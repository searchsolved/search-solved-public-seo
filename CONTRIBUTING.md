# Contributing

Thanks for your interest in contributing to this SEO tools collection!

## How to Contribute

### Reporting Bugs

- Open an [issue](https://github.com/searchsolved/search-solved-public-seo/issues) with a clear description
- Include steps to reproduce, expected vs actual behavior, and your Python version

### Suggesting Tools or Features

- Open an issue describing the tool or feature
- Explain the SEO use case it addresses

### Submitting a Pull Request

1. Fork the repo
2. Create a feature branch (`git checkout -b my-new-tool`)
3. Follow the structure of existing tools:
   - Place the tool in the appropriate category folder
   - Include a `requirements.txt`
   - Add a `README.md` with usage instructions
   - Use Streamlit for interactive tools where possible
4. Commit your changes
5. Open a PR with a clear description of what the tool does

### Code Guidelines

- **No API keys or credentials** — use environment variables or Streamlit input fields
- **No hardcoded paths** — make tools portable
- **Include dependencies** — every tool folder should have a `requirements.txt`
- **Keep it focused** — one tool per folder, one clear purpose per script

## Questions?

Open an issue or reach out on [LinkedIn](https://www.linkedin.com/in/lee-foot/) or [Bluesky](https://bsky.app/profile/leefootseo.bsky.social).
