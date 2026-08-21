{
  description = "German vocabulary extraction pipeline for DVDs and EPUBs (MakeMKV/ffmpeg -> faster-whisper -> spaCy)";

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
        devShells.default = pkgs.mkShell ({
          packages = [
            python
            pkgs.ffmpeg  # audio extraction from MakeMKV rips (ffmpeg + ffprobe)
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

            # GPU on WSL2: the Windows NVIDIA driver exposes the GPU as
            # /dev/dxg and libcuda under /usr/lib/wsl/lib. CTranslate2
            # additionally needs cuBLAS/cuDNN, which come as pip wheels
            # (the "gpu" extra).
            if [ -e /dev/dxg ] && [ -d /usr/lib/wsl/lib ]; then
              export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
              if ! python -c "import nvidia.cudnn" >/dev/null 2>&1; then
                echo "NVIDIA GPU detected — installing cuBLAS/cuDNN wheels (first run only) ..."
                python -m pip install -e ".[gpu]"
              fi
              cuda_libs=$(python -c "import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ':' + os.path.dirname(nvidia.cudnn.lib.__file__))" 2>/dev/null || true)
              if [ -n "$cuda_libs" ]; then
                export LD_LIBRARY_PATH="$cuda_libs:$LD_LIBRARY_PATH"
              fi
            fi

            # Apple Silicon: mlx-whisper runs Whisper on the M-series GPU
            # via Metal — several times faster than CPU inference. The
            # transcribe stage auto-selects it when importable.
            if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
              if ! python -c "import mlx_whisper" >/dev/null 2>&1; then
                echo "Apple Silicon detected — installing mlx-whisper (first run only) ..."
                python -m pip install -e ".[mlx]"
              fi
            fi
          '';
        } // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          # Manylinux wheels (ctranslate2, PyAV) need libstdc++/zlib from the
          # Nix store when running under the Nix-provided Python. macOS
          # wheels bundle their own dylibs, so this is Linux-only.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];
        });
      });
}
