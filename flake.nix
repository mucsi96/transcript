{
  description = "German DVD vocabulary extraction pipeline (VLC -> faster-whisper -> spaCy)";

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
        # Hybrid setup: Nix provides the system dependencies (Python, VLC
        # with libdvdcss, lsdvd); the Python dependencies live in a pip venv
        # because faster-whisper and the spaCy German models don't package
        # cleanly in nixpkgs.
        devShells.default = pkgs.mkShell {
          packages = [
            python
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.vlc    # includes libdvdcss for encrypted DVDs
            pkgs.lsdvd  # list DVD titles: lsdvd /dev/sr0
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
            if ! python -c "import transcript" >/dev/null 2>&1; then
              echo 'Next steps:'
              echo '  pip install -e ".[dev]"'
              echo '  python -m spacy download de_core_news_lg'
            fi
          '';
        };
      });
}
