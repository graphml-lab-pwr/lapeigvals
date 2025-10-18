all: quality
check_dirs := hallucinations scripts tests
# Check that source code meets quality standards

quality:
	uv run pre-commit run --all-files
	uv run mypy --install-types --non-interactive $(check_dirs)

fix:
	uv run pre-commit run --all-files

test:
	uv run pytest

install_cpu:
install_macos:
	uv sync --locked

install_cpu_dev:
	uv sync --locked --dev

install_gpu:
	uv sync
	uv pip install --no-build-isolation flash-attn==2.6.3

install_gpu_dev:
	uv sync  --locked --dev
	uv pip install --no-build-isolation flash-attn==2.6.3
