ما یک سرویس OCR ساده داریم که باید به یک Document Intelligence و Document Reconstruction Service کامل تبدیل شود.

هدف نهایی:
ورودی:
- PDF
- Image

خروجی:
1. فایل Markdown استاندارد
2. فایل Microsoft Word (.docx) با ساختار واقعی و قابل ویرایش

کیفیت خروجی باید به شکلی باشد که کاربر احساس کند سند اصلی بازسازی شده است.

========================================
Architecture
========================================

Pipeline باید به مراحل مستقل تقسیم شود:

1. Document Loader
2. Page Renderer
3. Layout Detection
4. Element Extraction
5. OCR/VLM Processing
6. Reading Order Resolver
7. Document AST Builder
8. Markdown Renderer
9. Word Renderer
10. Asset Manager


========================================
1. Document Loader
========================================

مسئولیت:

- دریافت فایل
- تشخیص نوع فایل
- استخراج صفحات
- ایجاد document_id

برای PDF:

هر صفحه باید با DPI بالا به Image تبدیل شود.

Metadata:

Document:
{
 id,
 filename,
 pages_count,
 created_at
}


Page:
{
 id,
 page_number,
 image_path,
 width,
 height
}


========================================
2. Layout Detection
========================================

از PP-DocLayout-L / PP-StructureV3 استفاده شود.

مدل فقط مسئول تشخیص ساختار است.
OCR نباید داخل Layout Detection باشد.


کلاس‌های Layout:

- title
- heading
- header
- footer
- paragraph
- list
- table
- table_caption
- table_footnote
- figure
- figure_caption
- chart
- formula
- code
- reference


خروجی:

LayoutElement:

{
 id,
 page_id,

 type,

 bbox:{
    x1,
    y1,
    x2,
    y2
 },

 confidence,

 crop_path
}


هر element باید crop شود.


مهم:

برای crop کردن:

bbox خام استفاده نشود.

10 درصد padding اضافه شود:

مثلا:

x1 -= width*0.05
x2 += width*0.05


========================================
3. Element Processing
========================================


هر element بر اساس type پردازش شود.


------------------
Text Elements
------------------

شامل:

- title
- heading
- header
- footer
- paragraph
- list
- reference


عملیات:

crop image

↓

OCR/VLM


خروجی:

TextElement:

{
 type,
 text,
 confidence
}



------------------
Table
------------------

Table را به صورت مستقل پردازش کن.


ورودی:

table crop


ارسال به VLM:

"Extract this table preserving rows and columns.
Return structured JSON."


خروجی:

{
 type:"table",

 rows:[
   []
 ]
}


بعداً:

Markdown:

|A|B|
|-|-|

Word:

ساخت Table واقعی DOCX


------------------
Formula
------------------

فرمول نباید به عنوان متن ساده ذخیره شود.


ورودی:

formula crop


VLM باید خروجی بدهد:

LaTeX


مثال:

\frac{x^2}{y}


Storage:

{
 type:"formula",
 latex:"..."
}


Renderer:

Markdown:

$$
\frac{x^2}{y}
$$


Word:

باید به Equation واقعی Word تبدیل شود.
از Office MathML / OMML استفاده کن.

فرمول نباید به صورت image داخل Word قرار گیرد.


------------------
Figure / Image
------------------

شامل:

- عکس
- illustration
- diagram


عملیات:

تصویر crop شده باید در Asset Manager ذخیره شود.


Asset:

{
 id,
 path,
 type:"image"
}


Markdown:

![description](assets/image_001.png)


Word:

Insert Image


همچنین VLM باید caption تولید کند:

{
 caption,
 description
}


------------------
Chart
------------------

Chart با image فرق دارد.


ورودی:

chart crop


VLM:

Extract:

- chart type
- title
- labels
- values
- explanation


خروجی:

{
 type:"chart",
 image:"...",
 description:"",
 data:[]
}


Markdown:

تصویر + توضیح


Word:

تصویر + caption


اگر امکان داشت chart data هم ذخیره شود.


========================================
4. Reading Order
========================================


بعد از استخراج تمام elementها:

Reading Order Resolver اجرا شود.


ورودی:

تمام bbox های صفحه


خروجی:

sequence:

[
 element_id_1,
 element_id_2,
 ...
]


قواعد:

برای فارسی:

- RTL support
- ستون راست قبل از ستون چپ
- داخل هر ستون بالا به پایین


برای انگلیسی:

LTR


Language detection لازم است.


========================================
5. Document AST
========================================


قبل از تولید خروجی یک مدل داخلی بساز.


مثال:


DocumentAST:

Page

 ├── Heading

 ├── Paragraph

 ├── Image

 ├── Table

 ├── Formula

 └── Chart


این AST منبع تولید همه خروجی‌ها باشد.


========================================
6. Markdown Renderer
========================================


Markdown باید:

- تصاویر را لینک کند
- فرمول‌ها را LaTeX کند
- جدول‌ها را Markdown Table کند


مثال:


# عنوان


متن پاراگراف


![chart](assets/chart_001.png)


$$
E=mc^2
$$


| ستون۱ | ستون۲ |
|---|---|
| مقدار | مقدار |



========================================
7. Word Renderer
========================================


خروجی DOCX باید واقعی باشد.


Rules:


Text:

→ Paragraph واقعی Word


Heading:

→ Word Heading styles


Table:

→ Word Table object


Image:

→ Insert image


Formula:

→ OMML Equation


Header/Footer:

→ Word Header/Footer section


List:

→ Word numbering/bullet list


Do NOT create Word by converting Markdown.


DOCX باید با python-docx یا کتابخانه مناسب مستقیماً ساخته شود.


========================================
8. Asset Manager
========================================


تمام تصاویر باید جدا ذخیره شوند.


ساختار:


output/

 document.md

 document.docx

 assets/

   image_001.png

   chart_001.png

   figure_001.png


Markdown باید فقط relative link بدهد.


========================================
Implementation Rules
========================================


- Layout و OCR کاملاً جدا باشند.
- همه intermediate resultها ذخیره شوند.
- هر مرحله قابل debug باشد.
- confidence تمام مدل‌ها ذخیره شود.
- مدل‌ها قابل تعویض باشند.
- GPU processing پشتیبانی شود.
- async queue استفاده شود.


========================================
Logging
========================================


برای هر صفحه:

ثبت شود:

- layout time
- OCR time
- VLM time
- render time


برای هر element:

ثبت شود:

- detected class
- confidence
- processing result


========================================
Final Goal
========================================


یک PDF فارسی پیچیده شامل:

- متن
- جدول
- فرمول
- نمودار
- عکس
- چند ستون
- header/footer


باید تبدیل شود به:

1. Markdown تمیز با asset link و LaTeX
2. Word واقعی با objectهای قابل ویرایش


Pipeline باید production quality باشد و فقط یک OCR wrapper نباشد.