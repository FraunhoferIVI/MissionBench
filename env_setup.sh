#!/usr/bin/env bash
set -euo pipefail
 
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
 
print_header() {
    echo
    echo "=============================================================="
    echo "$1"
    echo "=============================================================="
}
 
check_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv not found. Install it with:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo "Using uv: $(uv --version)"
}
 
setup_server() {
    print_header "Setting up server environment"
    uv sync --extra server --no-install-project
    echo "✅ Server dependencies installed"
 
    # Validate
    uv run python -c "import cosysairsim; import gymnasium; print('Server env OK')"
}
 
setup_missionbench() {
    print_header "Setting up missionbench environment"
    uv sync --extra missionbench --no-install-project
    echo "✅ MissionBench dependencies installed"
 
    # Validate
    uv run python -c "import torch; import pandas; import openai; print('MissionBench env OK')"
}
 
setup_all() {
    print_header "Setting up all dependencies"
    uv sync --all-extras --no-install-project
    echo "✅ All dependencies installed"
}
 
usage() {
    cat <<'EOF'
Usage:
    ./env_setup.sh                 # setup both envs (all extras)
    ./env_setup.sh server          # setup only server deps
    ./env_setup.sh missionbench    # setup only missionbench deps
 
All environments share a single .venv managed by uv.
EOF
}
 
main() {
    check_uv
 
    case "${1:-all}" in
        all)        setup_all ;;
        server)     setup_server ;;
        missionbench) setup_missionbench ;;
        -h|--help)  usage ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
 
    print_header "Setup complete"
    echo "Run commands with: uv run python <script>"
    echo "Or activate:       source .venv/bin/activate"
}
 
main "$@"
 