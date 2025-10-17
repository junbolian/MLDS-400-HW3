# src/r/install_packages.R
# Install minimal, stable packages for this HW
pkgs <- c(
  "readr",     # fast CSV reader
  "dplyr",     # data wrangling
  "stringr",   # string helpers
  "tibble"     # tibbles
)
inst <- installed.packages()[, "Package"]
to_install <- setdiff(pkgs, inst)
if (length(to_install)) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}
