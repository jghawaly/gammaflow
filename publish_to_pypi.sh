#!/bin/bash
# GammaFlow PyPI Publishing Script
# ================================
# This script guides you through publishing GammaFlow to PyPI

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                    GAMMAFLOW PyPI PUBLISHING SCRIPT"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo

# Step 1: Pre-flight checks
echo -e "${YELLOW}STEP 1: Pre-flight Checks${NC}"
echo "───────────────────────────"

# Check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}❌ Virtual environment not activated!${NC}"
    echo "Run: source venv/bin/activate"
    exit 1
else
    echo -e "${GREEN}✅ Virtual environment active${NC}"
fi

# Check if tests pass
echo "Running tests..."
if pytest tests/ -v --tb=short; then
    echo -e "${GREEN}✅ All tests pass${NC}"
else
    echo -e "${RED}❌ Tests failed! Fix them before publishing.${NC}"
    exit 1
fi

# Check for TODO items in setup.py
if grep -q "TODO" setup.py; then
    echo -e "${YELLOW}⚠️  WARNING: setup.py contains TODO items${NC}"
    grep "TODO" setup.py
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo

# Step 2: Install build tools
echo -e "${YELLOW}STEP 2: Install Build Tools${NC}"
echo "──────────────────────────"
pip install --upgrade build twine
echo -e "${GREEN}✅ Build tools installed${NC}"
echo

# Step 3: Clean previous builds
echo -e "${YELLOW}STEP 3: Clean Previous Builds${NC}"
echo "─────────────────────────────"
rm -rf build/ dist/ *.egg-info
echo -e "${GREEN}✅ Cleaned build directories${NC}"
echo

# Step 4: Build package
echo -e "${YELLOW}STEP 4: Build Package${NC}"
echo "────────────────────"
python -m build
echo -e "${GREEN}✅ Package built successfully${NC}"
echo "   Created:"
ls -lh dist/
echo

# Step 5: Check package
echo -e "${YELLOW}STEP 5: Check Package${NC}"
echo "────────────────────"
twine check dist/*
echo -e "${GREEN}✅ Package passes checks${NC}"
echo

# Step 6: Choose upload target
echo -e "${YELLOW}STEP 6: Upload Package${NC}"
echo "─────────────────────"
echo "Choose upload target:"
echo "  1) Test PyPI (recommended first)"
echo "  2) Production PyPI"
echo "  3) Skip upload"
read -p "Choice (1-3): " choice

case $choice in
    1)
        echo
        echo -e "${YELLOW}Uploading to Test PyPI...${NC}"
        echo "You'll need your Test PyPI credentials"
        echo "(Register at: https://test.pypi.org/account/register/)"
        echo
        twine upload --repository testpypi dist/*
        echo
        echo -e "${GREEN}✅ Uploaded to Test PyPI!${NC}"
        echo
        echo "Test installation:"
        echo "  pip install --index-url https://test.pypi.org/simple/ gammaflow"
        echo
        echo "View at: https://test.pypi.org/project/gammaflow/"
        ;;
    2)
        echo
        echo -e "${RED}⚠️  UPLOADING TO PRODUCTION PyPI${NC}"
        echo "This will make your package publicly available!"
        read -p "Are you sure? (type 'yes'): " confirm
        if [ "$confirm" = "yes" ]; then
            echo
            echo -e "${YELLOW}Uploading to PyPI...${NC}"
            echo "You'll need your PyPI credentials"
            echo "(Register at: https://pypi.org/account/register/)"
            echo
            twine upload dist/*
            echo
            echo -e "${GREEN}✅ Uploaded to PyPI!${NC}"
            echo
            echo "Anyone can now install with:"
            echo "  pip install gammaflow"
            echo
            echo "View at: https://pypi.org/project/gammaflow/"
            echo
            echo -e "${YELLOW}Don't forget to:${NC}"
            echo "  • Create git tag: git tag v0.1.0 && git push --tags"
            echo "  • Create GitHub release with release notes"
            echo "  • Announce your package!"
        else
            echo "Upload cancelled."
        fi
        ;;
    3)
        echo "Skipping upload. Distribution files are in dist/"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo
echo "═══════════════════════════════════════════════════════════════════════════════"
echo -e "${GREEN}                         PUBLISHING COMPLETE!${NC}"
echo "═══════════════════════════════════════════════════════════════════════════════"

