{
  description = "German audio transcript learner using Speechmatics + GPT-5";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        python = pkgs.python312;

        # Native libraries the Python wheels link against at runtime.
        # PortAudio backs `sounddevice`; PulseAudio is how WSLg exposes the
        # audio server so we can tap the system-audio monitor source.
        runtimeLibs = with pkgs; [
          portaudio
          libpulseaudio
          stdenv.cc.cc.lib
          zlib
        ];

        libraryPath = pkgs.lib.makeLibraryPath runtimeLibs;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.pulseaudio # provides `pactl`/`parec` for listing audio sources
          ] ++ runtimeLibs;

          # Wheels such as sounddevice dlopen() the native libs above, so make
          # sure they can be found regardless of how Python was launched.
          LD_LIBRARY_PATH = libraryPath;
          PYTHONBREAKPOINT = "0";

          shellHook = ''
            export LD_LIBRARY_PATH="${libraryPath}:''${LD_LIBRARY_PATH:-}"

            if [ ! -d .venv ]; then
              echo "Creating virtualenv (.venv) ..."
              uv venv --python ${python}/bin/python .venv
            fi
            # shellcheck disable=SC1091
            source .venv/bin/activate

            echo "Syncing Python dependencies ..."
            uv pip install -e . >/dev/null

            if ! python -c "import de_core_news_sm" >/dev/null 2>&1; then
              echo "Downloading spaCy German model (de_core_news_sm) ..."
              python -m spacy download de_core_news_sm >/dev/null || \
                echo "  (model download failed — lemmatization will be skipped)"
            fi

            echo ""
            echo "Ready. Configure .env (see .env.example), then run:"
            echo "    transcript-learner        # or: python -m transcript_learner"
            echo ""
          '';
        };

        # `nix run` convenience wrapper — assumes deps were synced via the
        # devShell at least once (it reuses ./.venv).
        apps.default = {
          type = "app";
          program = toString (pkgs.writeShellScript "transcript-learner" ''
            export LD_LIBRARY_PATH="${libraryPath}:''${LD_LIBRARY_PATH:-}"
            if [ ! -d .venv ]; then
              echo "No .venv found. Enter the dev shell first: nix develop" >&2
              exit 1
            fi
            source .venv/bin/activate
            exec python -m transcript_learner "$@"
          '');
        };
      });
}
