"""
Embed real TrueType font files into a generated .docx package.

Word only ever renders text using whatever font is actually installed on
the *viewing* device -- "IRNazanin" (this pipeline's default Persian
complex-script font) is a common Iranian system font but is never
guaranteed to be present on every device the output is opened on. When
it's missing, Word silently substitutes a fallback font for that run,
and a fallback font without proper Arabic/Persian OpenType shaping
tables can misbehave in ways well beyond a wrong glyph -- including,
per real-world reports, disrupting the run's effective layout direction.

Embedding the real font file removes that dependency entirely. This
follows the exact ODTTF font-embedding mechanism real Microsoft Word
itself uses (ECMA-376 Part 1, font embedding): the first 32 bytes of the
font file are XORed with a 16-byte key, and that key is derived from a
GUID -- generated fresh per embedded font -- by reversing the GUID's raw
16 bytes. The (non-reversed) GUID string is stored in the ``w:fontKey``
attribute so a reader can re-derive the same key and reverse the XOR.

This works as raw zip/XML post-processing on an already-rendered .docx,
rather than through python-docx's object model, because python-docx has
no API for adding arbitrary new package parts (font binaries) or
relationships.
"""

from __future__ import annotations

import functools
import logging
import uuid
import zipfile
from io import BytesIO
from pathlib import Path

from lxml import etree

logger = logging.getLogger(__name__)

_FONTS_DIR = Path(__file__).parent / "fonts"

# Font family name (matching w:rFonts usage elsewhere in this renderer) ->
# style -> bundled .ttf file. Only fonts listed here are ever embedded --
# render_docx falls back to other complex-script fonts (e.g. one detected
# from the source PDF) when the document doesn't use one of these, and
# those aren't bundled/embeddable.
BUNDLED_FONTS: dict[str, dict[str, Path]] = {
    "IRNazanin": {
        "regular": _FONTS_DIR / "IRNazanin-Regular.ttf",
        "bold": _FONTS_DIR / "IRNazanin-Bold.ttf",
    },
}

_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_PKG_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_FONT_REL_TYPE = f"{_R_NS}/font"
_OBFUSCATED_FONT_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.obfuscatedFont"
)

# CT_Font's embed* children, in the order the schema requires them among
# themselves (regular, bold, italic, bold+italic).
_STYLE_TAGS = {
    "regular": "embedRegular",
    "bold": "embedBold",
    "italic": "embedItalic",
    "boldItalic": "embedBoldItalic",
}
_EMBED_TAG_ORDER = ["embedRegular", "embedBold", "embedItalic", "embedBoldItalic"]

FontStyles = dict[str, bytes]  # style name (see _STYLE_TAGS) -> raw .ttf bytes


def _obfuscate(font_bytes: bytes) -> tuple[bytes, str]:
    """Return (obfuscated_bytes, fontKey_attr_value) for one font file."""
    guid = uuid.uuid4()
    key = guid.bytes[::-1]
    head = bytearray(font_bytes[:32])
    for i in range(len(head)):
        head[i] ^= key[i % 16]
    obfuscated = bytes(head) + font_bytes[32:]
    font_key = "{" + str(guid).upper() + "}"
    return obfuscated, font_key


def _q(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def _load_or_create_rels(in_zip: zipfile.ZipFile, rels_path: str) -> etree._Element:
    if rels_path in in_zip.namelist():
        return etree.fromstring(in_zip.read(rels_path))
    root = etree.Element(_q(_PKG_RELS_NS, "Relationships"))
    return root


def _next_rel_id(rels: etree._Element) -> str:
    existing = {el.get("Id") for el in rels.findall(_q(_PKG_RELS_NS, "Relationship"))}
    n = 1
    while f"rId{n}" in existing:
        n += 1
    return f"rId{n}"


def _ensure_fntdata_content_type(content_types: etree._Element) -> None:
    already = content_types.xpath(
        './ct:Default[@Extension="fntdata"]', namespaces={"ct": _CT_NS}
    )
    if already:
        return
    default_el = etree.SubElement(content_types, _q(_CT_NS, "Default"))
    default_el.set("Extension", "fntdata")
    default_el.set("ContentType", _OBFUSCATED_FONT_CONTENT_TYPE)


def _find_or_create_font_element(
    font_table: etree._Element, font_name: str
) -> etree._Element:
    for candidate in font_table.findall(_q(_W_NS, "font")):
        if candidate.get(_q(_W_NS, "name")) == font_name:
            return candidate
    font_el = etree.SubElement(font_table, _q(_W_NS, "font"))
    font_el.set(_q(_W_NS, "name"), font_name)
    return font_el


def _reorder_embed_children(font_el: etree._Element) -> None:
    """Keep CT_Font's embedRegular/Bold/Italic/BoldItalic in schema order."""
    by_tag = {}
    for child in list(font_el):
        local = etree.QName(child).localname
        if local in _EMBED_TAG_ORDER:
            by_tag[local] = child
            font_el.remove(child)
    for tag_name in _EMBED_TAG_ORDER:
        if tag_name in by_tag:
            font_el.append(by_tag[tag_name])


def _enable_embed_true_type_fonts(settings: etree._Element) -> None:
    """Set <w:embedTrueTypeFonts w:val="1"/>, right after <w:zoom/> per schema."""
    if settings.find(_q(_W_NS, "embedTrueTypeFonts")) is not None:
        return
    el = etree.Element(_q(_W_NS, "embedTrueTypeFonts"))
    el.set(_q(_W_NS, "val"), "1")
    zoom = settings.find(_q(_W_NS, "zoom"))
    if zoom is not None:
        zoom.addnext(el)
    else:
        settings.insert(0, el)


def embed_fonts(docx_bytes: bytes, fonts: dict[str, FontStyles]) -> bytes:
    """
    Embed real TrueType font files into a rendered .docx package.

    ``fonts`` maps a font family name (matching the name used in the
    document's ``w:rFonts``) to its available styles, e.g.
    ``{"IRNazanin": {"regular": ttf_bytes, "bold": bold_ttf_bytes}}``.
    Returns the new .docx bytes; the input bytes are not modified.
    """
    in_zip = zipfile.ZipFile(BytesIO(docx_bytes))

    content_types = etree.fromstring(in_zip.read("[Content_Types].xml"))
    font_table = etree.fromstring(in_zip.read("word/fontTable.xml"))
    settings = etree.fromstring(in_zip.read("word/settings.xml"))
    rels_path = "word/_rels/fontTable.xml.rels"
    rels = _load_or_create_rels(in_zip, rels_path)

    _ensure_fntdata_content_type(content_types)
    _enable_embed_true_type_fonts(settings)

    new_parts: dict[str, bytes] = {}
    font_index = 1
    for font_name, styles in fonts.items():
        font_el = _find_or_create_font_element(font_table, font_name)
        for style, ttf_bytes in styles.items():
            obfuscated, font_key = _obfuscate(ttf_bytes)
            part_name = f"word/fonts/font{font_index}.fntdata"
            target = f"fonts/font{font_index}.fntdata"
            font_index += 1
            new_parts[part_name] = obfuscated

            rid = _next_rel_id(rels)
            rel_el = etree.SubElement(rels, _q(_PKG_RELS_NS, "Relationship"))
            rel_el.set("Id", rid)
            rel_el.set("Type", _FONT_REL_TYPE)
            rel_el.set("Target", target)

            tag = _STYLE_TAGS[style]
            existing_embed = font_el.find(_q(_W_NS, tag))
            if existing_embed is not None:
                font_el.remove(existing_embed)
            embed_el = etree.SubElement(font_el, _q(_W_NS, tag))
            embed_el.set(_q(_R_NS, "id"), rid)
            embed_el.set(_q(_W_NS, "fontKey"), font_key)
        _reorder_embed_children(font_el)

    def _serialize(el: etree._Element) -> bytes:
        return etree.tostring(
            el, xml_declaration=True, encoding="UTF-8", standalone=True
        )

    replacements = {
        "[Content_Types].xml": content_types,
        "word/fontTable.xml": font_table,
        "word/settings.xml": settings,
    }

    out_buf = BytesIO()
    with zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as out_zip:
        for item in in_zip.infolist():
            if item.filename in replacements:
                data = _serialize(replacements[item.filename])
            elif item.filename == rels_path:
                continue  # written below, in either case
            else:
                data = in_zip.read(item.filename)
            out_zip.writestr(item.filename, data)
        out_zip.writestr(rels_path, _serialize(rels))
        for part_name, part_bytes in new_parts.items():
            out_zip.writestr(part_name, part_bytes)

    return out_buf.getvalue()


@functools.cache
def _read_bundled_font(path: Path) -> bytes:
    return path.read_bytes()


def embed_bundled_fonts(docx_bytes: bytes, font_names: set[str]) -> bytes:
    """
    Embed whichever of ``font_names`` this module ships a real .ttf for.

    Silently skips names not in BUNDLED_FONTS (e.g. a font detected from
    the source PDF instead of the IRNazanin default) rather than failing
    the whole render -- font embedding is a quality improvement, not a
    correctness requirement.
    """
    fonts: dict[str, FontStyles] = {}
    for name in font_names:
        styles = BUNDLED_FONTS.get(name)
        if not styles:
            continue
        try:
            fonts[name] = {
                style: _read_bundled_font(path) for style, path in styles.items()
            }
        except OSError:
            logger.warning("Bundled font file missing for %r, skipping embed", name)
    if not fonts:
        return docx_bytes
    return embed_fonts(docx_bytes, fonts)
