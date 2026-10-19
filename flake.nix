{
  description = "neoracer-mujoco: nix develop gives Python 3.15 with the package installed editable, and Node for js/";

  # The nixos-unstable channel, as a tarball: it has Python 3.15.
  inputs.nixpkgs.url = "https://channels.nixos.org/nixos-unstable/nixexprs.tar.xz";

  outputs =
    { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAll (
        pkgs:
        let
          python = pkgs.python315;
          # What the PyPI wheels (mujoco, numpy, scipy, pygame-ce) load at run
          # time on Linux: the C++ runtime, OpenGL and GLFW for the viewer, and
          # the X11 and Wayland libraries GLFW opens a window with.
          libs = with pkgs; [
            stdenv.cc.cc.lib
            zlib
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
          # The Python packages come from PyPI into .venv, with uv, not from
          # nixpkgs. nixpkgs has MuJoCo 3.16.0 for Python 3.15, but its binary
          # cache has no Python 3.15 packages yet, so the first `nix develop`
          # would build numpy, scipy and MuJoCo from source, and its etils
          # needs jax for its tests, which has no Python 3.15 build. Once
          # nixpkgs caches them, python.withPackages can replace .venv.
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
              pkgs.nodejs_22
              pkgs.git-lfs
            ];
            env = {
              UV_PYTHON_DOWNLOADS = "never";
            } // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
            };
            # .venv on nixpkgs' Python 3.15, with the package installed
            # editable: an edit to src/ takes effect on the next run.
            shellHook = ''
              [ -d .venv ] || uv venv -q --python ${python}/bin/python3.15 .venv
              uv pip install -q --python .venv/bin/python -e ".[dev,demo]"
              . .venv/bin/activate
            '';
          };
        }
      );
    };
}
