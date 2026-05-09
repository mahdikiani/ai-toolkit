"""Routes for prompt management."""

from fastapi import APIRouter, HTTPException, Request

from server.config import Settings

from .schemas import PromptListResponse, PromptSchemaResponse

router = APIRouter(prefix="/prompts", tags=["Prompts"])


@router.get("/", response_model=list[PromptListResponse])
async def list_prompts(request: Request) -> list[PromptListResponse]:
    """List all available prompts."""
    prompts_dir = Settings.prompts_dir
    prompts: list[PromptListResponse] = []

    if not prompts_dir.exists():
        return prompts

    for prompt_file in prompts_dir.glob("*.yaml"):
        if prompt_file.stem.startswith("_"):
            continue

        prompts.append(
            PromptListResponse(
                name=prompt_file.stem,
                description=f"Prompt template: {prompt_file.stem}",
                tags=[],
            )
        )

    return prompts


@router.get("/{prompt_name}/schema", response_model=PromptSchemaResponse)
async def get_prompt_schema(request: Request, prompt_name: str) -> PromptSchemaResponse:
    """Get the schema for a specific prompt."""
    prompts_dir = Settings.prompts_dir
    prompt_path = prompts_dir / f"{prompt_name}.yaml"

    if not prompt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Prompt '{prompt_name}' not found",
        )

    # TODO: Parse the YAML file and extract input fields and output schema
    # For now, return basic info
    return PromptSchemaResponse(
        name=prompt_name,
        description=f"Prompt template: {prompt_name}",
        tags=[],
        input_fields=[],
        output_schema=None,
    )
