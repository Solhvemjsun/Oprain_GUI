{
  description = "Flake for Solysis develop";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells = {
          default = pkgs.mkShell {
            nativeBuildInputs = with pkgs; [
              (python3.withPackages (
                python-pkgs: with python-pkgs; [
                  # Solysis
                  requests
                  numpy
                  tkinter
                  jupyterlab
                  matplotlib
                  pandas
                  scipy
                  # Brainflow https://brainflow.readthedocs.io/en/stable/Examples.html
                  brainflow
                  pygame
                ]
              ))
              # pkgs.texlive.combined.scheme-full
            ];
            shellHook = ''
              echo "Thanks for using Solysis, happy research!"
              # jupyter-lab
            '';
          };

        };
      }
    );
}
