FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS fast-uv

WORKDIR /app

COPY pyproject.toml pyproject.toml

# RUN uv sync --no-dev --no-cache --no-install-project
RUN uv pip install --system --no-cache-dir --upgrade .


FROM python:3.12-slim AS fast-server

WORKDIR /app

RUN apt-get update \
    && apt-get install -y libmagic1 poppler-utils ffmpeg \
        libcairo2 libffi-dev libgdk-pixbuf-2.0-0 libharfbuzz-subset0 \
        libpango-1.0-0 libpangocairo-1.0-0 libpangoft2-1.0-0 \
        shared-mime-info fonts-liberation2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages from build stage
COPY --from=fast-uv /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages

RUN addgroup --system --gid 1000 user \
    && adduser --system --uid 1000 --ingroup user --disabled-password --gecos '' --home /home/user user \
    # && mkdir -p /home/user \
    # && chown -R user:user /home/user \
    && mkdir -p /app/logs \
    && chown -R user:user /app/logs

USER user
# Copy application code
COPY --chown=user:user . .

CMD [ "python", "main.py" ]
