{
  description = "German DVD vocabulary extraction pipeline (MakeMKV/ffmpeg -> faster-whisper -> spaCy)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python311;
      in
      {
        # Hybrid setup: Nix provides the system dependencies (Python,
        # ffmpeg); the Python dependencies live in a pip venv because
        # faster-whisper and the spaCy German models don't package cleanly
        # in nixpkgs.
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.ffmpeg  # audio extraction from MakeMKV rips (ffmpeg + ffprobe)
          ];

          # Manylinux wheels (ctranslate2, PyAV) need libstdc++/zlib from the
          # Nix store when running under the Nix-provided Python.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];

          shellHook = ''
            if [ ! -d .venv ]; then
              echo "Creating virtualenv in .venv ..."
              ${python.interpreter} -m venv .venv
            fi
            source .venv/bin/activate
            if ! python -m pip show transcript >/dev/null 2>&1; then
              echo "Installing Python dependencies (first run only) ..."
              python -m pip install -e ".[dev]"
            fi
            if ! python -c "import de_core_news_lg" >/dev/null 2>&1; then
              echo "Downloading spaCy model de_core_news_lg (~570 MB, first run only) ..."
              python -m spacy download de_core_news_lg
            fi
          '';
        };
      });
}
