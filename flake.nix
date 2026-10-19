{
  description = "neoracer-mujoco: nix develop gives Python 3.15 with the package installed editable, and Node for js/";

  inputs = {
    # The nixos-unstable channel, as a tarball: it has Python 3.15.
    nixpkgs.url = "https://channels.nixos.org/nixos-unstable/nixexprs.tar.xz";
    # uv2nix builds the Python environment from uv.lock.
    pyproject-nix = {
      url = "git+https://github.com/pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "git+https://github.com/pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "git+https://github.com/pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      # MuJoCo has no wheel for Intel macOS.
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAll = f: lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
    in
    {
      devShells = forAll (
        pkgs:
        let
          # The packages are uv.lock's wheels from PyPI, built into the Nix
          # store. nixpkgs' own Python 3.15 packages are not in its binary
          # cache yet, so the first nix develop would build numpy, scipy and
          # MuJoCo from source; once they are, python315.withPackages could
          # replace uv.lock.
          pythonSet =
            (pkgs.callPackage pyproject-nix.build.packages { python = pkgs.python315; }).overrideScope
              (
                lib.composeManyExtensions [
                  pyproject-build-systems.overlays.wheel
                  (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
                  # The package itself from the checkout, so an edit to src/
                  # takes effect on the next run.
                  (workspace.mkEditablePyprojectOverlay { root = "$REPO_ROOT"; })
                  (final: prev: {
                    neoracer-mujoco = prev.neoracer-mujoco.overrideAttrs (old: {
                      nativeBuildInputs = old.nativeBuildInputs ++ final.resolveBuildSystem { editables = [ ]; };
                    });
                  })
                ]
              );
          # The jax extra is left out: jaxlib has no Python 3.15 wheels.
          python = pythonSet.mkVirtualEnv "neoracer-mujoco-dev" {
            neoracer-mujoco = [
              "dev"
              "demo"
            ];
          };
          # What the wheels load at run time on Linux: OpenGL and GLFW for the
          # viewer, and the X11 and Wayland libraries GLFW opens a window with.
          libs = with pkgs; [
            libGL
            glfw
            libxkbcommon
            wayland
            libx11
            libxcursor
            libxi
            libxinerama
            libxrandr
            libxext
          ];
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
              pkgs.nodejs_22
              pkgs.git-lfs
            ];
            env = {
              # uv locks (uv lock) but never installs: Nix does.
              UV_NO_SYNC = "1";
              UV_PYTHON = "${python}/bin/python";
              UV_PYTHON_DOWNLOADS = "never";
            }
            // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              LD_LIBRARY_PATH = lib.makeLibraryPath libs;
            };
            # The editable package is the checkout nix develop runs in, so it
            # must run inside one.
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
              if [ ! -f "$REPO_ROOT/src/neoracer_mujoco/__init__.py" ]; then
                echo "nix develop: run it inside the neoracer-mujoco checkout" >&2
                exit 1
              fi
            '';
          };
        }
      );
    };
}
