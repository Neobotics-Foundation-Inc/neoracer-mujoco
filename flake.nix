{
  description = "neoracer-mujoco: nix develop gives Python 3.14 with pyproject.toml's dependencies from nixpkgs, the package from src/, and Node for js/";

  inputs = {
    # The nixos-unstable channel, as a tarball. Its binary cache has every
    # Python 3.14 package pyproject.toml names, so nix develop builds nothing.
    nixpkgs.url = "https://channels.nixos.org/nixos-unstable/nixexprs.tar.xz";
    # pyproject.nix reads pyproject.toml and finds each dependency in nixpkgs.
    pyproject-nix = {
      url = "git+https://github.com/pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, pyproject-nix, ... }:
    let
      inherit (nixpkgs) lib;
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAll = f: lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
      project = pyproject-nix.lib.project.loadPyproject { projectRoot = ./.; };
    in
    {
      devShells = forAll (
        pkgs:
        let
          python = pkgs.python314.override {
            packageOverrides = final: prev: {
              # pyproject.toml pins 3.0.0, which takes several modules in one
              # run; nixpkgs has 2.5.5.
              pybind11-stubgen = prev.pybind11-stubgen.overridePythonAttrs rec {
                version = "3.0.0";
                src = pkgs.fetchPypi {
                  pname = "pybind11_stubgen";
                  inherit version;
                  hash = "sha256-sKkIrpkgTiAaUUjBfcPLeJAhozmrncHQUXojqHSQIEw=";
                };
              };
            };
          };
          # pyproject.toml's dependencies and its dev, demo and jax extras.
          # jax runs on the CPU here: a GPU needs jax's CUDA build.
          extras = [
            "dev"
            "demo"
            "jax"
          ];
          env = python.withPackages (project.renderers.withPackages { inherit python extras; });
          # The nixpkgs versions that break a pin in pyproject.toml, such as
          # mujoco==3.16.0 after a nixpkgs update.
          conflicts = lib.mapAttrs (_: c: c.version) (
            project.validators.validateVersionConstraints { inherit python extras; }
          );
          # What MuJoCo's viewer loads at run time on Linux: OpenGL and GLFW,
          # and the X11 and Wayland libraries GLFW opens a window with.
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
          default =
            assert lib.assertMsg (
              conflicts == { }
            ) "nixpkgs does not meet pyproject.toml's pins: ${builtins.toJSON conflicts}";
            pkgs.mkShell {
              packages = [
                env
                pkgs.nodejs_22
                pkgs.git
                pkgs.git-lfs
              ];
              env = lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
                LD_LIBRARY_PATH = lib.makeLibraryPath libs;
              };
              # The package is the checkout's src/, put on the path rather than
              # installed: an edit takes effect the next time Python starts.
              shellHook = ''
                REPO_ROOT=$(${pkgs.git}/bin/git rev-parse --show-toplevel 2>/dev/null)
                if [ ! -f "$REPO_ROOT/src/neoracer_mujoco/__init__.py" ]; then
                  echo "nix develop: run it inside the neoracer-mujoco checkout" >&2
                  exit 1
                fi
                export PYTHONPATH="$REPO_ROOT/src"
              '';
            };
        }
      );
    };
}
