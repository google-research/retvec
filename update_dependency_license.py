"""Generates the DEPENDENCY_LICENSES file, by extracting and classifying dependencies."""
import setuptools
import pkg_resources
import re
import tabulate
import os
from setup import read, get_version

METADATA_FILES = ["METADATA", "PKG-INFO"]

LICENSE_OVERRIDES = {}


LICENSE_CATEGORIES = {
    "Apache 2.0": "notice",
    "Apache 2.0 OR MIT": "notice",
    "HPND": "notice",
    "BSD": "notice",
    "MIT": "notice",
    "PSF": "notice",
    "MPLv2.0, MIT": "reciprocal, notice",
    "Public domain": "unencumbered",
}


def GetDependencies():
    """Extracts dependencies from setup.py"""
    dependencies = []

    def setup(**kwargs):
        dependencies.extend(kwargs["install_requires"])

    setuptools.setup = setup
    content = open("setup.py").read()
    exec(content)
    return dependencies


def GetPackageLicenses(package_name):
    """Extract the licensing metadata from a Python package."""
    packages = pkg_resources.require(package_name)
    package = packages[0]
    for metadata_file in METADATA_FILES:
        if not package.has_metadata(metadata_file):
            continue
        metadata = package.get_metadata(metadata_file)
        if package_name in LICENSE_OVERRIDES:
            license = LICENSE_OVERRIDES[package_name]
        else:
            match = re.search("^License:\s*(.*)$", metadata,
                              re.MULTILINE | re.IGNORECASE)
            license = re.sub(
                "\sLicences|\sLicense,?|Version\s", "", match.group(1))
            license = re.sub("-", " ", license)
        match = re.search("^Home-page:\s*(.*)$", metadata,
                          re.MULTILINE | re.IGNORECASE)
        homepage = match.group(1)
        return license, homepage
    return None


def GenerateDependencyLicensesFile():
    """Generates the DEPENDENCY_LICENSES file."""
    license_data = []
    for package_name in GetDependencies():
        license, homepage = GetPackageLicenses(package_name)
        license_category = LICENSE_CATEGORIES.get(license, "")
        license_data.append(
            (package_name, license, license_category, homepage))
    table = tabulate.tabulate(
        sorted(license_data, key=lambda x: x[1]),
        headers=["Package", "License", "Category", "Homepage"],
    )
    print(table)
    with open("DEPENDENCY_LICENSES", "w") as f:
        f.write(table)


def Main():
    GenerateDependencyLicensesFile()


if __name__ == "__main__":
    Main()
