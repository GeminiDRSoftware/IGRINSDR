# Maintenance Guide

## Version Management

This project uses `setuptools-scm` for automatic version management based on git tags. The version number is automatically derived from git tags and made available as `igrinsdr.__version__`.

### Version Format
- The version follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`
- Development versions are marked with a `.devN` suffix
- Pre-releases use `a` (alpha), `b` (beta), or `rc` (release candidate) suffixes

### Setting a New Version

To create a new release:

1. Make sure all changes are committed
2. Create an annotated tag with the new version (prefixed with 'v'):
   ```bash
   git tag -a v1.2.3 -m "Release v1.2.3"
   ```
3. Push the tag to the remote repository:
   ```bash
   git push origin v1.2.3
   ```

### Development Versions

- During development between releases, the version will automatically include a `.devN` suffix and git hash
- Example: `1.2.3.dev4+gabc1234` means it's the 4th commit after the 1.2.3 tag

### Accessing Version Information

In Python:
```python
import igrinsdr
print(igrinsdr.__version__)          # e.g., '1.2.3'
print(igrinsdr.__version_tuple__)    # e.g., (1, 2, 3)
```

### Troubleshooting

- If you get `0.0.0` as the version, make sure:
  - `setuptools-scm` is installed in your build environment
  - You're working in a git repository with at least one tag
  - The package is installed in development mode or properly built

### Dependencies

- `setuptools-scm` is required for version management
- The version is automatically generated during package build/installation

## Doc Management

- To convert user's manual to rst:
```sh
quarto render IGRINSDR_users_manual.ipynb --to rst
```

