# LangExtract (Lite)

A lightweight Streamlit web application for extracting structured information from PDF documents using Large Language Models (OpenAI GPT or Google Gemini).

## 🚀 Features

- **PDF Processing**: Upload PDFs or specify file paths, with support for encrypted PDFs
- **Flexible Extraction**: Customizable schema prompts for targeted information extraction
- **Multi-Model Support**: Choose between OpenAI GPT models or Google Gemini models
- **Parallel Processing**: Configurable chunking and parallel processing for faster extraction
- **Progress Tracking**: Real-time progress bars and status updates
- **Natural Language Summaries**: AI-generated summaries of extracted data
- **Highlighted Context**: Visual highlighting of extracted text in original document
- **Performance Optimization**: Preset configurations for different file sizes
- **Session Persistence**: Remembers user preferences and API keys across sessions

## 📋 Requirements

- Python 3.8+
- OpenAI API key or Google API key
- PDF files to process

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/WMatt-Barnes/LangExtract.git
   cd LangExtract
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys:**
   - Create a `.env` file in the project root
   - Add your API keys:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     # OR
     GOOGLE_API_KEY=your_google_api_key_here
     ```

## 🚀 Usage

1. **Run the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the displayed URL (usually `http://localhost:8501`)

3. **Configure the app:**
   - Select your preferred model provider (OpenAI or Gemini)
   - Choose a specific model
   - Enter or update your API key if needed

4. **Process a PDF:**
   - Upload a PDF file or provide a file path
   - Enter your extraction schema/prompt (e.g., "extract assembly steps, torque values, and required tools")
   - Adjust performance settings if needed
   - Click "Run Extraction"

5. **Review results:**
   - View structured extraction results in a table
   - Read the AI-generated natural language summary
   - Explore highlighted context in the original document
   - Check extraction statistics and breakdowns

## ⚙️ Configuration

### Performance Settings

- **Small files**: Optimized for documents under 50 pages
- **Standard files**: Balanced settings for typical documents
- **Large files**: Conservative settings for very large documents

### Custom Settings

- **Chunk size**: Number of characters per processing chunk
- **Max workers**: Parallel processing threads
- **Timeout**: API request timeout in seconds
- **Max tokens**: Maximum output tokens from the model

## 🔧 API Providers

### OpenAI
- **Models**: gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo
- **Rate limits**: ~3-5 requests/minute (free tier)
- **Best for**: High-quality extractions, complex schemas

### Google Gemini
- **Models**: gemini-1.5-flash, gemini-1.5-pro, gemini-pro
- **Rate limits**: ~15 requests/minute (free tier)
- **Best for**: Cost-effective processing, faster responses

## 📊 Output Format

The app extracts structured data in the following format:

```json
{
  "extractions": [
    {
      "extraction_class": "string",
      "extraction_text": "verbatim text from document",
      "attributes": {
        "key": "value"
      },
      "evidence": ["supporting quotes"],
      "pages": [1, 2, 3]
    }
  ]
}
```

## 🎯 Use Cases

- **Technical Documentation**: Extract procedures, specifications, and requirements
- **Research Papers**: Identify key findings, methodologies, and conclusions
- **Legal Documents**: Find clauses, terms, and obligations
- **Medical Records**: Extract diagnoses, treatments, and outcomes
- **Financial Reports**: Identify metrics, trends, and insights

## 🚨 Troubleshooting

### Common Issues

1. **PDF Encryption**: Install `pycryptodome` for AES-encrypted PDFs
2. **API Rate Limits**: Reduce parallel workers or increase chunk size
3. **JSON Parsing Errors**: The app includes automatic repair mechanisms
4. **Memory Issues**: Limit page count or reduce chunk size for very large documents

### Performance Tips

- **Small documents**: Use larger chunk sizes and fewer workers
- **Large documents**: Use smaller chunks and more workers
- **Rate-limited APIs**: Reduce worker count to stay within limits

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by OpenAI GPT and Google Gemini models
- PDF processing with [PyPDF2](https://pypdf2.readthedocs.io/)

## 📞 Support

For issues, questions, or feature requests, please open an issue on GitHub or contact the maintainers.
