import os
import io
import re
import json
import html
import random
import concurrent.futures
from typing import Dict, List, Tuple, Any

import streamlit as st
import pandas as pd
try:
    from dotenv import load_dotenv, find_dotenv, set_key
except ImportError:
    # Fallback for environments where dotenv might not be available
    def load_dotenv(*args, **kwargs):
        pass
    def find_dotenv(*args, **kwargs):
        return None
    def set_key(*args, **kwargs):
        pass
from PyPDF2 import PdfReader


def initialize_environment() -> None:
    load_dotenv()
    st.set_page_config(page_title="LangExtract (Lite)", layout="wide")


def save_to_local_storage(key: str, value: str) -> None:
    """Save a value to browser localStorage using JavaScript"""
    js_code = f"""
    <script>
        if (typeof(Storage) !== "undefined") {{
            localStorage.setItem("{key}", "{value}");
            console.log("Saved to localStorage:", "{key}", "{value}");
        }}
    </script>
    """
    st.components.v1.html(js_code, height=0)


def load_from_local_storage(key: str, default: str = "") -> str:
    """Load a value from browser localStorage using JavaScript"""
    js_code = f"""
    <script>
        if (typeof(Storage) !== "undefined") {{
            const value = localStorage.getItem("{key}");
            if (value !== null) {{
                console.log("Loaded from localStorage:", "{key}", value);
                // Store in sessionStorage for immediate access
                sessionStorage.setItem("{key}", value);
            }}
        }}
    </script>
    """
    st.components.v1.html(js_code, height=0)
    return default


def get_persistent_value(key: str, default: str = "") -> str:
    """Get a value that persists across sessions using session state and localStorage"""
    # First check session state (current session)
    if key in st.session_state:
        return st.session_state[key]
    
    # Then check if we have a stored value in session state
    stored_key = f"stored_{key}"
    if stored_key in st.session_state:
        return st.session_state[stored_key]
    
    # Try to load from localStorage
    stored_value = load_from_local_storage(key, default)
    if stored_value != default:
        # Store in session state for this session
        st.session_state[stored_key] = stored_value
        return stored_value
    
    return default


def set_persistent_value(key: str, value: str) -> None:
    """Set a value that persists across sessions"""
    # Store in current session
    st.session_state[key] = value
    
    # Store for future sessions
    st.session_state[f"stored_{key}"] = value
    
    # Save to localStorage
    save_to_local_storage(key, value)


def read_pdf_pages(uploaded_file: io.BytesIO, password: str | None = None) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    try:
        reader = PdfReader(uploaded_file)
    except Exception as exc:
        # Handle missing crypto backend for AES-encrypted PDFs
        if "PyCryptodome is required" in str(exc):
            raise RuntimeError(
                "This PDF uses AES encryption. Install pycryptodome and retry: "
                ".\\.venv\\Scripts\\python -m pip install pycryptodome"
            )
        raise
    try:
        if reader.is_encrypted:
            if password:
                result = reader.decrypt(password)
            else:
                # Try empty password first when none provided
                result = reader.decrypt("")
            if isinstance(result, int) and result == 0:
                raise RuntimeError("Incorrect PDF password or decryption failed.")
    except Exception as exc:
        raise RuntimeError(f"PDF appears encrypted and requires a password: {exc}")
    page_index: int
    for page_index in range(len(reader.pages)):
        page_text: str = reader.pages[page_index].extract_text() or ""
        pages.append({"page": page_index + 1, "text": page_text})
    return pages


def chunk_text(pages: List[Dict[str, Any]], max_chars: int = 15000) -> List[Tuple[str, List[int]]]:
    chunks: List[Tuple[str, List[int]]] = []
    current_buffer: List[str] = []
    current_pages: List[int] = []
    current_length: int = 0
    for page in pages:
        page_text: str = page["text"]
        page_len: int = len(page_text)
        if current_length + page_len > max_chars and current_buffer:
            chunks.append(("\n".join(current_buffer), current_pages.copy()))
            current_buffer = []
            current_pages = []
            current_length = 0
        current_buffer.append(page_text)
        current_pages.append(page["page"])
        current_length += page_len
    if current_buffer:
        chunks.append(("\n".join(current_buffer), current_pages.copy()))
    return chunks


def get_provider_and_model() -> Tuple[str, str]:
    provider: str = os.getenv("LANGEXTRACT_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "gemini"}:
        provider = "openai"
    if provider == "openai":
        model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    else:
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return provider, model


def _ensure_env_file() -> str:
    try:
        env_path = find_dotenv(usecwd=True)
        if not env_path:
            env_path = os.path.join(os.getcwd(), ".env")
        if not os.path.exists(env_path):
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("")
        return env_path
    except Exception:
        # Fallback for environments where .env file operations aren't needed
        return os.path.join(os.getcwd(), ".env")


def call_openai(model: str, system_prompt: str, user_prompt: str, timeout_s: int | None = None, max_tokens: int | None = None, force_json: bool = False) -> str:
    from openai import OpenAI

    client = OpenAI()
    
    # Prepare the API call parameters
    api_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "timeout": timeout_s,
        "max_tokens": max_tokens,
    }
    
    # Only add response_format for JSON responses
    if force_json:
        api_params["response_format"] = {"type": "json_object"}
    
    completion = client.chat.completions.create(**api_params)
    return completion.choices[0].message.content or ""


def call_gemini(model: str, system_prompt: str, user_prompt: str, timeout_s: int | None = None, max_tokens: int | None = None) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
    generation_config = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "max_output_tokens": max_tokens or 512,
    }
    model_ref = genai.GenerativeModel(model_name=model, generation_config=generation_config)
    prompt = f"{system_prompt}\n\n{user_prompt}"
    response = model_ref.generate_content(prompt)
    return response.text or ""


def extract_json_block(text: str) -> str:
    pattern = re.compile(r"\{[\s\S]*\}\s*$")
    match = pattern.search(text.strip())
    if match:
        return match.group(0)
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    # Best-effort: take substring from first '{' to last '}'
    try:
        s = text.strip()
        i = s.find('{')
        j = s.rfind('}')
        if 0 <= i < j:
            return s[i:j + 1]
    except Exception:
        pass
    # Last resort: try to find any JSON-like structure
    try:
        # Look for patterns like "extraction_class": "value"
        if '"extraction_class"' in text and '"extraction_text"' in text:
            start = text.find('"extraction_class"')
            if start >= 0:
                # Find the opening brace before this
                brace_start = text.rfind('{', 0, start)
                if brace_start >= 0:
                    # Find the closing brace after
                    brace_end = text.find('}', start)
                    if brace_end > brace_start:
                        return text[brace_start:brace_end + 1]
    except Exception:
        pass
    return text


def repair_truncated_json(json_text: str) -> str:
    """Attempt to repair truncated JSON by completing incomplete objects"""
    try:
        # If it's already valid JSON, return as-is
        json.loads(json_text)
        return json_text
    except json.JSONDecodeError as e:
        # Handle various truncation patterns
        if "Expecting ',' delimiter" in str(e) or "Expecting property name" in str(e) or "Expecting value" in str(e):
            # Try to find the last complete object and close it properly
            lines = json_text.split('\n')
            repaired_lines = []
            brace_count = 0
            in_extractions = False
            last_complete_line = -1
            
            for i, line in enumerate(lines):
                if '"extractions"' in line:
                    in_extractions = True
                if '{' in line:
                    brace_count += line.count('{')
                if '}' in line:
                    brace_count -= line.count('}')
                
                # Track the last line that looks complete
                if line.strip().endswith('}') or line.strip().endswith(']') or line.strip().endswith(','):
                    last_complete_line = i
                
                # If we're in extractions and hit an incomplete object, try to close it
                if in_extractions and brace_count == 1 and line.strip().endswith(','):
                    # This looks like an incomplete object, close it
                    repaired_lines.append(line.rstrip(','))
                    repaired_lines.append('    }')
                    brace_count -= 1
                else:
                    repaired_lines.append(line)
            
            # If we have an incomplete last object, try to complete it
            if last_complete_line >= 0 and last_complete_line < len(lines) - 1:
                # Look for the last incomplete object
                for i in range(last_complete_line + 1, len(lines)):
                    line = lines[i].strip()
                    if line and not line.endswith('}') and not line.endswith(']') and not line.endswith(','):
                        # This line is incomplete, try to complete it
                        if '"extraction_text"' in line and not line.endswith('"'):
                            # Complete the extraction_text value
                            repaired_lines.append(line + '"')
                        elif '"evidence"' in line and not line.endswith(']'):
                            # Complete the evidence array
                            repaired_lines.append(line + ']')
                        elif '"attributes"' in line and not line.endswith('}'):
                            # Complete the attributes object
                            repaired_lines.append(line + '}')
                        else:
                            # Just close the object
                            repaired_lines.append(line + '}')
                        break
            
            # Close any remaining open braces
            while brace_count > 0:
                repaired_lines.append('  }')
                brace_count -= 1
            
            repaired_text = '\n'.join(repaired_lines)
            try:
                json.loads(repaired_text)
                return repaired_text
            except:
                pass
        
        # Additional repair attempt: look for common truncation patterns
        try:
            # If the text ends with incomplete quotes or brackets, try to complete them
            if json_text.strip().endswith('"'):
                # Ends with quote, might need comma and closing brace
                if '"extraction_text"' in json_text and '"evidence"' not in json_text:
                    # Missing evidence array
                    json_text += ', "evidence": []'
                if '"attributes"' in json_text and '"evidence"' not in json_text:
                    # Missing evidence array
                    json_text += ', "evidence": []'
                if '"extraction_class"' in json_text and '"attributes"' not in json_text:
                    # Missing attributes object
                    json_text += ', "attributes": {}'
                # Close the object
                json_text += '}'
                # Close the array
                json_text += ']}'
                # Close the main object
                json_text += '}'
                
                try:
                    json.loads(json_text)
                    return json_text
                except:
                    pass
        except:
            pass
        
        return json_text


def get_preset_settings(preset: str) -> Dict[str, int]:
    # Heuristic suggestions
    if preset == "large":
        return {"max_pages": 10, "chunk_chars": 8000, "timeout_s": 120, "max_tokens": 256}
    elif preset == "standard":
        return {"max_pages": 25, "chunk_chars": 12000, "timeout_s": 60, "max_tokens": 512}
    elif preset == "small":
        return {"max_pages": 0, "chunk_chars": 15000, "timeout_s": 60, "max_tokens": 512}
    else:
        return {"max_pages": 0, "chunk_chars": 15000, "timeout_s": 60, "max_tokens": 512}


def process_single_chunk(provider: str, model: str, system_prompt: str, instruction: str, chunk_data: Tuple[str, List[int]], timeout_s: int, max_tokens: int) -> List[Dict[str, Any]]:
    """Process a single chunk and return extracted items"""
    chunk_text_value, chunk_pages = chunk_data
    items = []
    
    user_prompt = (
        f"DOCUMENT PAGES: {chunk_pages}\n\n"
        f"DOCUMENT TEXT:\n{chunk_text_value}\n\n"
        f"{instruction}\n"
        "Return only JSON."
    )
    
    # 1st attempt
    try:
        if provider == "openai":
            raw = call_openai(model, system_prompt, user_prompt, timeout_s=timeout_s, max_tokens=max_tokens, force_json=True)
        else:
            raw = call_gemini(model, system_prompt, user_prompt, timeout_s=timeout_s, max_tokens=max_tokens)
    except Exception as exc:
        st.error(f"Model call failed for chunk {chunk_pages}: {exc}")
        return items

    parsed_ok = False
    for attempt in range(2):  # parse, then one retry if needed
        try:
            json_text = extract_json_block(raw)
            # Try to repair truncated JSON
            json_text = repair_truncated_json(json_text)
            data = json.loads(json_text)
            chunk_items = data.get("extractions", [])
            for item in chunk_items:
                item_copy = {
                    "extraction_class": str(item.get("extraction_class", "")).strip(),
                    "extraction_text": str(item.get("extraction_text", "")).strip(),
                    "attributes": item.get("attributes", {}),
                    "evidence": item.get("evidence", []),
                    "pages": chunk_pages,
                }
                items.append(item_copy)
            parsed_ok = True
            break
        except Exception as exc:
            if attempt == 0:
                strict_suffix = (
                    "\n\nReturn ONLY a valid JSON object with top-level 'extractions'. "
                    "Do not include text before/after. Ensure the JSON is complete and not truncated."
                )
                try:
                    if provider == "openai":
                        raw = call_openai(
                            model, system_prompt, user_prompt + strict_suffix,
                            timeout_s=timeout_s, max_tokens=max_tokens, force_json=True,
                        )
                    else:
                        raw = call_gemini(
                            model, system_prompt, user_prompt + strict_suffix,
                            timeout_s=timeout_s, max_tokens=max_tokens,
                        )
                except Exception as exc:
                    st.error(f"Retry model call failed for chunk {chunk_pages}: {exc}")
                    break
            else:
                break

    if not parsed_ok:
        # Final attempt with repair
        try:
            json_text = extract_json_block(raw)
            repaired_json = repair_truncated_json(json_text)
            if repaired_json != json_text:
                # Try to parse the repaired version
                try:
                    data = json.loads(repaired_json)
                    chunk_items = data.get("extractions", [])
                    for item in chunk_items:
                        item_copy = {
                            "extraction_class": str(item.get("extraction_class", "")).strip(),
                            "extraction_text": str(item.get("extraction_text", "")).strip(),
                            "attributes": item.get("attributes", {}),
                            "evidence": item.get("evidence", []),
                            "pages": chunk_pages,
                        }
                        items.append(item_copy)
                except:
                    pass
        except Exception as e:
            st.error(f"Final parsing attempt failed for chunk {chunk_pages}: {e}")
    
    return items


def run_extraction(provider: str, model: str, schema_prompt: str, chunks: List[Tuple[str, List[int]]], timeout_s: int, max_tokens: int, max_workers: int) -> Tuple[List[Dict[str, Any]], bool]:
    system_prompt = (
        "You are LangExtract (lite). Extract structured facts from technical PDFs. "
        "Return a STRICT JSON object only. No prose, no code fences."
    )
    instruction = (
        "Extract according to this schema description: "
        f"\n\nSCHEMA:\n{schema_prompt}\n\n"
        "Return a single JSON object with a top-level key 'extractions' which is a list of items. "
        "Each item must include: \n"
        "- extraction_class: string\n"
        "- extraction_text: string (verbatim phrase from the text)\n"
        "- attributes: object (key-value pairs derived from schema)\n"
        "- evidence: list of short quotes from the provided text\n"
        "Limit to concise, high-confidence items."
    )

    all_items: List[Dict[str, Any]] = []
    total_chunks = len(chunks)
    was_stopped = False
    
    # Create progress bar for overall extraction
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Determine optimal concurrency based on chunk count
    max_workers = min(max_workers, total_chunks)  # Don't exceed chunk count
    
    if total_chunks > 1:
        st.info(f"Processing {total_chunks} chunks with {max_workers} parallel workers for faster extraction")
        
        # Use parallel processing for multiple chunks
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(process_single_chunk, provider, model, system_prompt, instruction, chunk, timeout_s, max_tokens): chunk 
                for chunk in chunks
            }
            
            # Process completed chunks as they finish
            completed_chunks = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_items = future.result()
                    all_items.extend(chunk_items)
                    completed_chunks += 1
                    
                    # Update progress
                    progress = completed_chunks / total_chunks
                    progress_bar.progress(progress)
                    status_text.text(f"Completed {completed_chunks} of {total_chunks} chunks ({len(all_items)} extractions found)")
                    
                    # Check for stop request
                    if st.session_state.get("stop_requested"):
                        was_stopped = True
                        st.info(f"🛑 **Stop requested!** Completed {completed_chunks}/{total_chunks} chunks. Showing partial results.")
                        # Cancel remaining futures
                        for f in future_to_chunk:
                            f.cancel()
                        break
                        
                except Exception as exc:
                    st.error(f"Chunk {chunk[1]} generated an exception: {exc}")
                    completed_chunks += 1
                    progress = completed_chunks / total_chunks
                    progress_bar.progress(progress)
    else:
        # Single chunk - process directly
        status_text.text("Processing single chunk...")
        all_items = process_single_chunk(provider, model, system_prompt, instruction, chunks[0], timeout_s, max_tokens)
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
    
    # Clear progress indicators when done
    progress_bar.empty()
    status_text.empty()
    
    # Show final status
    if was_stopped:
        st.warning(f"🛑 **Extraction stopped early** - Completed {len(chunks)} chunks, found {len(all_items)} extractions")
    else:
        st.success(f"✅ **Extraction complete** - Processed all {len(chunks)} chunks, found {len(all_items)} extractions")
    
    return all_items, was_stopped


def suggest_performance_settings(total_pages: int, total_chars: int) -> Dict[str, int]:
    # Heuristic suggestions
    if total_pages >= 200 or total_chars >= 800_000:
        return {"max_pages": 10, "chunk_chars": 8000, "timeout_s": 120, "max_tokens": 256}
    if total_pages >= 80 or total_chars >= 300_000:
        return {"max_pages": 15, "chunk_chars": 10000, "timeout_s": 90, "max_tokens": 384}
    if total_pages >= 30 or total_chars >= 120_000:
        return {"max_pages": 25, "chunk_chars": 12000, "timeout_s": 60, "max_tokens": 512}
    return {"max_pages": 0, "chunk_chars": 15000, "timeout_s": 60, "max_tokens": 512}


def optimize_chunk_size(pages: List[Dict[str, Any]], target_chunks: int = 4) -> int:
    """Optimize chunk size to get approximately target_chunks chunks"""
    if not pages:
        return 15000
    
    total_chars = sum(len(p.get("text", "")) for p in pages)
    if total_chars == 0:
        return 15000
    
    # Calculate optimal chunk size to get target_chunks
    optimal_size = total_chars // target_chunks
    
    # Ensure reasonable bounds
    optimal_size = max(5000, min(optimal_size, 25000))
    
    return optimal_size


def estimate_processing_time(chunks: List[Tuple[str, List[int]]], max_workers: int, timeout_s: int) -> str:
    """Estimate total processing time"""
    if not chunks:
        return "Unknown"
    
    total_chunks = len(chunks)
    avg_time_per_chunk = timeout_s * 0.7  # Assume 70% of timeout on average
    
    if max_workers >= total_chunks:
        # All chunks processed simultaneously
        estimated_time = avg_time_per_chunk
    else:
        # Sequential processing with parallel workers
        estimated_time = (total_chunks / max_workers) * avg_time_per_chunk
    
    if estimated_time < 60:
        return f"~{estimated_time:.0f} seconds"
    else:
        minutes = estimated_time / 60
        return f"~{minutes:.1f} minutes"


def build_results_table(extractions: List[Dict[str, Any]], pages: List[Dict[str, Any]]) -> pd.DataFrame:
    def find_source_citation(extraction_text: str) -> str:
        normalized_query = extraction_text.strip().lower()
        if not normalized_query:
            return ""
        for page in pages:
            page_text = page["text"] or ""
            idx = page_text.lower().find(normalized_query)
            if idx >= 0:
                window_start = max(0, idx - 80)
                window_end = min(len(page_text), idx + len(normalized_query) + 80)
                snippet = page_text[window_start:window_end].replace("\n", " ")
                return f"p{page['page']}: …{snippet}…"
        return ""

    rows: List[Dict[str, Any]] = []
    for item in extractions:
        row = {
            "extraction_class": item.get("extraction_class", ""),
            "extraction_text": item.get("extraction_text", ""),
            "attributes": json.dumps(item.get("attributes", {}), ensure_ascii=False),
            "source_citation": find_source_citation(item.get("extraction_text", "")),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=["extraction_class", "extraction_text", "attributes", "source_citation"])


def compute_highlight_spans(extractions: List[Dict[str, Any]], pages: List[Dict[str, Any]]) -> Dict[int, List[Tuple[int, int, Dict[str, Any]]]]:
    page_to_spans: Dict[int, List[Tuple[int, int, Dict[str, Any]]]] = {}
    for item in extractions:
        query = (item.get("extraction_text") or "").strip()
        if not query:
            continue
        normalized_query = query.lower()
        for page in pages:
            text = page["text"] or ""
            search_space = text.lower()
            start = 0
            while True:
                idx = search_space.find(normalized_query, start)
                if idx < 0:
                    break
                end = idx + len(query)
                page_to_spans.setdefault(page["page"], []).append((idx, end, item))
                start = end
    for page, spans in page_to_spans.items():
        spans.sort(key=lambda t: t[0])
    return page_to_spans


def render_highlighted_text(pages: List[Dict[str, Any]], page_to_spans: Dict[int, List[Tuple[int, int, Dict[str, Any]]]]) -> None:
    class_to_color: Dict[str, str] = {}
    color_pool: List[str] = [
        "#fff3b0",
        "#c1fba4",
        "#a0e7e5",
        "#bdb2ff",
        "#ffd6a5",
        "#ffadad",
    ]
    color_index: int = 0

    st.markdown(
        "<style> .lex-highlight { padding: 1px 2px; border-radius: 3px; } .lex-page { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9rem; } .lex-page h4 { margin-top: 0; } </style>",
        unsafe_allow_html=True,
    )

    for page in pages:
        page_num = page["page"]
        text = page["text"] or ""
        spans = page_to_spans.get(page_num, [])
        if not spans:
            continue
        html_parts: List[str] = []
        cursor: int = 0
        for start, end, item in spans:
            safe_chunk = html.escape(text[cursor:start])
            html_parts.append(safe_chunk)
            extraction_class = item.get("extraction_class", "")
            if extraction_class not in class_to_color:
                class_to_color[extraction_class] = color_pool[color_index % len(color_pool)]
                color_index += 1
            color = class_to_color[extraction_class]
            attrs_preview = html.escape(json.dumps(item.get("attributes", {}), ensure_ascii=False))
            highlight_text = html.escape(text[start:end])
            tooltip = html.escape(f"{extraction_class} | {attrs_preview}")
            html_parts.append(
                f"<span class='lex-highlight' style='background-color:{color}' title='{tooltip}'>" + highlight_text + "</span>"
            )
            cursor = end
        html_parts.append(html.escape(text[cursor:]))
        with st.expander(f"Page {page_num} context"):
            st.markdown(f"<div class='lex-page'>{''.join(html_parts)}</div>", unsafe_allow_html=True)


def generate_extraction_summary(provider: str, model: str, schema_prompt: str, extractions: List[Dict[str, Any]], timeout_s: int, max_tokens: int) -> str:
    """Generate a natural language summary of the extracted data"""
    if not extractions:
        return "No extractions found to summarize."
    
    # Prepare the extracted data for the LLM
    extraction_summary = []
    for i, item in enumerate(extractions, 1):
        extraction_class = item.get("extraction_class", "Unknown")
        extraction_text = item.get("extraction_text", "")
        attributes = item.get("attributes", {})
        evidence = item.get("evidence", [])
        pages = item.get("pages", [])
        
        summary_item = f"{i}. **{extraction_class}**: {extraction_text}"
        if attributes:
            attr_str = ", ".join([f"{k}: {v}" for k, v in attributes.items()])
            summary_item += f" (Attributes: {attr_str})"
        if evidence:
            summary_item += f" - Found on pages {pages}"
        extraction_summary.append(summary_item)
    
    system_prompt = "You are an expert technical analyst who provides clear, actionable insights from technical documents."
    summary_prompt = f"""
Based on the extracted data below, provide a comprehensive natural language summary that answers the user's original query.

ORIGINAL QUERY: {schema_prompt}

EXTRACTED DATA:
{chr(10).join(extraction_summary)}

Please provide a structured summary with the following sections:

**EXECUTIVE SUMMARY**
- A concise overview of what was found and its significance

**DETAILED FINDINGS**
- List each extraction in natural language, organized by category
- Include relevant attributes and page references
- Use clear, professional language

**KEY INSIGHTS & PATTERNS**
- Identify common themes, trends, or relationships
- Highlight any notable patterns or exceptions

**PRACTICAL IMPLICATIONS**
- What this means for the user
- Recommendations or next steps
- Any critical considerations

**TECHNICAL DETAILS**
- Summary of pages covered
- Total number of extractions by category
- Any limitations or gaps in the data

Write in clear, professional language that a technical professional would understand. Focus on actionable insights and make the structured data easily digestible.
"""
    
    try:
        if provider == "openai":
            raw_summary = call_openai(model, system_prompt, summary_prompt, timeout_s=timeout_s, max_tokens=max_tokens, force_json=False)
        else:
            raw_summary = call_gemini(model, system_prompt, summary_prompt, timeout_s=timeout_s, max_tokens=max_tokens)
        
        # Clean up the response
        summary = raw_summary.strip()
        if summary.startswith("```") and summary.endswith("```"):
            summary = summary[3:-3].strip()
        if summary.startswith("markdown"):
            summary = summary[8:].strip()
        
        return summary
        
    except Exception as exc:
        return f"Could not generate summary due to error: {exc}"


def main() -> None:
    initialize_environment()

    st.title("LangExtract (Lite)")
    provider, model = get_provider_and_model()

    # Initialize session state with persistent values
    if "initialized" not in st.session_state:
        # Load all persistent values at startup
        st.session_state["max_pages"] = int(get_persistent_value("max_pages", "0"))
        st.session_state["chunk_chars"] = int(get_persistent_value("chunk_chars", "15000"))
        st.session_state["timeout_s"] = int(get_persistent_value("timeout_s", "60"))
        st.session_state["max_tokens"] = int(get_persistent_value("max_tokens", "512"))
        st.session_state["schema_prompt"] = get_persistent_value("schema_prompt", "extract assembly steps, torque values, and required tools")
        st.session_state["provider"] = get_persistent_value("provider", provider)
        st.session_state["initialized"] = True
    
    # Initialize other session state variables
    if "stop_requested" not in st.session_state:
        st.session_state["stop_requested"] = False
    if "change_openai_key" not in st.session_state:
        st.session_state["change_openai_key"] = False
    if "change_google_key" not in st.session_state:
        st.session_state["change_google_key"] = False

    with st.sidebar:
        # Debug section (can be removed in production)
        with st.expander("🔧 Debug Info", expanded=False):
            st.write("**Session State:**")
            for key, value in st.session_state.items():
                if "key" in key.lower() and value:
                    # Mask API keys for security
                    masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                    st.write(f"{key}: {masked_value}")
                else:
                    st.write(f"{key}: {value}")
            
            st.write("**Environment:**")
            st.write(f"Provider: {provider}")
            st.write(f"Model: {model}")
        
        st.markdown("**Model Provider**")
        provider_choice = st.selectbox(
            "Select Provider",
            options=["openai", "gemini"],
            index=0 if provider == "openai" else 1,
            key="provider_choice"
        )
        
        # Update provider if changed
        if provider_choice != provider:
            set_persistent_value("provider", provider_choice)
            st.session_state["provider"] = provider_choice
            st.rerun()
        
        st.markdown("**Model**")
        if provider == "openai":
            model_choice = st.selectbox(
                "OpenAI Model",
                options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                index=0 if model == "gpt-4o-mini" else 1,
                key="openai_model_choice"
            )
        else:
            model_choice = st.selectbox(
                "Gemini Model",
                options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
                index=0 if model == "gemini-1.5-flash" else 1,
                key="gemini_model_choice"
            )
        
        # Update model if changed
        if model_choice != model:
            if provider == "openai":
                set_persistent_value("openai_model", model_choice)
                os.environ["OPENAI_MODEL"] = model_choice
            else:
                set_persistent_value("gemini_model", model_choice)
                os.environ["GEMINI_MODEL"] = model_choice
            st.rerun()

        st.markdown("---")
        st.markdown("**API Key Management**")
        
        # Add clear keys option
        if st.button("🗑️ Clear All Stored Keys", type="secondary"):
            # Clear from session state
            for key in list(st.session_state.keys()):
                if "api_key" in key.lower() or "stored_" in key:
                    del st.session_state[key]
            # Clear from localStorage
            save_to_local_storage("openai_api_key", "")
            save_to_local_storage("google_api_key", "")
            st.success("All stored API keys cleared!")
            st.rerun()
        
        if provider == "openai":
            # Get persistent API key
            stored_key = get_persistent_value("openai_api_key", "")
            has_key = bool(stored_key)
            
            if has_key:
                st.success("✅ OpenAI API key configured")
                if st.button("🔄 Change OpenAI API Key"):
                    st.session_state["change_openai_key"] = True
                
                if st.session_state.get("change_openai_key", False):
                    new_key = st.text_input("New OpenAI API Key", type="password", key="new_openai_key")
                    remember_key = st.checkbox("Remember this key", value=True, key="remember_new_openai")
                    if new_key:
                        if remember_key:
                            set_persistent_value("openai_api_key", new_key)
                            st.success("OpenAI API key updated and saved permanently!")
                        else:
                            st.session_state["openai_api_key"] = new_key
                            st.success("OpenAI API key updated for this session only!")
                        os.environ["OPENAI_API_KEY"] = new_key
                        st.session_state["change_openai_key"] = False
                        st.rerun()
            else:
                st.info("Enter your OpenAI API key")
                key_in = st.text_input("OpenAI API Key", type="password", key="openai_key_input")
                remember_key = st.checkbox("Remember this key", value=True, key="remember_openai")
                if key_in:
                    if remember_key:
                        set_persistent_value("openai_api_key", key_in)
                        st.success("OpenAI API key saved and will be remembered!")
                    else:
                        st.session_state["openai_api_key"] = key_in
                        st.success("OpenAI API key saved for this session only!")
                    os.environ["OPENAI_API_KEY"] = key_in
                    st.rerun()
        else:
            # Get persistent API key
            stored_key = get_persistent_value("google_api_key", "")
            has_key = bool(stored_key)
            
            if has_key:
                st.success("✅ Google API key configured")
                if st.button("🔄 Change Google API Key"):
                    st.session_state["change_google_key"] = True
                
                if st.session_state.get("change_google_key", False):
                    new_key = st.text_input("New Google API Key", type="password", key="new_google_key")
                    remember_key = st.checkbox("Remember this key", value=True, key="remember_new_google")
                    if new_key:
                        if remember_key:
                            set_persistent_value("google_api_key", new_key)
                            st.success("Google API key updated and saved permanently!")
                        else:
                            st.session_state["google_api_key"] = new_key
                            st.success("Google API key updated for this session only!")
                        os.environ["GOOGLE_API_KEY"] = new_key
                        st.session_state["change_google_key"] = False
                        st.rerun()
            else:
                st.info("Enter your Google API key")
                key_in = st.text_input("Google API Key", type="password", key="google_key_input")
                remember_key = st.checkbox("Remember this key", value=True, key="remember_google")
                if key_in:
                    if remember_key:
                        set_persistent_value("google_api_key", key_in)
                        st.success("Google API key saved and will be remembered!")
                    else:
                        st.session_state["google_api_key"] = key_in
                        st.success("Google API key saved for this session only!")
                    os.environ["GOOGLE_API_KEY"] = key_in
                    st.rerun()

    source_mode = st.radio("PDF Source", options=["Upload", "File path"], horizontal=True)
    pdf_password = st.text_input("PDF password (optional)", type="password")
    with st.expander("Performance settings", expanded=False):
        # Preset buttons
        col1, col2, col3 = st.columns(3)
        if col1.button("Small files", use_container_width=True):
            preset = get_preset_settings("small")
            for k, v in preset.items():
                st.session_state[k] = v
                set_persistent_value(k, str(v))
            st.rerun()
        if col2.button("Standard files", use_container_width=True):
            preset = get_preset_settings("standard")
            for k, v in preset.items():
                st.session_state[k] = v
                set_persistent_value(k, str(v))
            st.rerun()
        if col3.button("Large files", use_container_width=True):
            preset = get_preset_settings("large")
            for k, v in preset.items():
                st.session_state[k] = v
                set_persistent_value(k, str(v))
            st.rerun()
        
        st.markdown("---")
        
        # Performance optimization controls
        st.markdown("**📈 Performance Optimizations**")
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            max_workers = st.number_input(
                "Max parallel workers", min_value=1, max_value=8, step=1,
                value=4, help="Number of chunks to process simultaneously. Higher = faster but may hit rate limits."
            )
            chunk_chars = st.number_input(
                "Chunk size (characters)", min_value=2000, step=1000,
                value=int(st.session_state.get("chunk_chars", 15000)), key="chunk_chars",
                help="Larger chunks = fewer API calls but may be slower per chunk"
            )
        
        with col_opt2:
            max_pages = st.number_input(
                "Max pages to read (0 = all)", min_value=0, step=1,
                value=int(st.session_state.get("max_pages", 0)), key="max_pages",
            )
            timeout_s = st.number_input(
                "Request timeout (seconds)", min_value=5, step=5,
                value=int(st.session_state.get("timeout_s", 60)), key="timeout_s",
            )
            max_tokens = st.number_input(
                "Max model tokens (output)", min_value=128, step=64,
                value=int(st.session_state.get("max_tokens", 512)), key="max_tokens",
            )
        
        # Performance tips
        st.markdown("""
        **💡 Performance Tips:**
        - **Speed up extraction**: Increase chunk size, use parallel workers, limit pages, optimize token limits
        - **Rate limits**: OpenAI ~3-5 req/min (free), Gemini ~15 req/min (free) - adjust workers accordingly
        """)
    uploaded_file = None
    path_input: str = ""
    if source_mode == "Upload":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    else:
        path_input = st.text_input("PDF file path", value="", placeholder=r"C:\\path\\to\\file.pdf")
    schema_prompt = st.text_area(
        "Schema / Prompt",
        value=st.session_state.get("schema_prompt", "extract assembly steps, torque values, and required tools"),
        height=120,
        key="schema_prompt",
        on_change=lambda: set_persistent_value("schema_prompt", st.session_state.get("schema_prompt", "")),
    )

    col_run, col_stop, col_rerun = st.columns([1, 1, 1])
    run = col_run.button("Run Extraction")
    if col_stop.button("Stop"):
        st.session_state["stop_requested"] = True
        st.info("Stop requested. It will take effect after the current chunk.")
    if col_rerun.button("Rerun"):
        st.session_state["stop_requested"] = False
        st.rerun()

    if run:
        # Clear any previous stop requests before starting
        st.session_state["stop_requested"] = False

        if source_mode == "Upload":
            if not uploaded_file:
                st.error("Please upload a PDF file.")
                return
            try:
                with st.spinner("Reading PDF…"):
                    pages = read_pdf_pages(uploaded_file, password=pdf_password or None)
            except Exception as exc:
                st.error(f"Failed to read PDF: {exc}")
                return
        else:
            if not path_input:
                st.error("Please provide a PDF file path.")
                return
            if not os.path.exists(path_input):
                st.error("File path does not exist.")
                return
            try:
                with open(path_input, "rb") as f:
                    with st.spinner("Reading PDF…"):
                        pages = read_pdf_pages(f, password=pdf_password or None)
            except Exception as exc:
                st.error(f"Failed to read PDF: {exc}")
                return

        if not any(p.get("text") for p in pages):
            st.error("Could not read text from PDF.")
            return

        # Show PDF reading results
        st.success(f"Successfully read {len(pages)} pages from PDF")
        
        if max_pages and max_pages > 0:
            pages = pages[:int(max_pages)]
            st.info(f"Limited to first {len(pages)} pages as per settings")
        
        chunks = chunk_text(pages, max_chars=int(chunk_chars))
        st.info(f"Split into {len(chunks)} chunks for processing")
        
        # Show optimization suggestions
        if len(chunks) > 1:
            optimal_size = optimize_chunk_size(pages, target_chunks=max_workers)
            if abs(optimal_size - chunk_chars) > 2000:
                st.info(f"💡 **Optimization tip**: Consider chunk size {optimal_size:,} for better parallel processing with {max_workers} workers")
            
            estimated_time = estimate_processing_time(chunks, max_workers, timeout_s)
            st.info(f"⏱️ **Estimated processing time**: {estimated_time}")

        with st.spinner("Calling model…"):
            extractions, was_stopped = run_extraction(provider, model, schema_prompt, chunks, int(timeout_s), int(max_tokens), max_workers)

        # Always show results section, even if no extractions
        st.subheader("📊 Structured Results")
        
        if not extractions:
            if was_stopped:
                st.info("No extractions found in completed chunks. Try running with different settings or check your schema prompt.")
            else:
                st.info("No extractions found. Try adjusting your schema prompt or chunk size.")
            return
        
        # Show completion status
        if was_stopped:
            st.warning(f"🛑 **Partial Results** - Extraction was stopped early. Showing {len(extractions)} extractions from completed chunks.")
            
            # Add resume option
            col_resume1, col_resume2 = st.columns([1, 3])
            with col_resume1:
                if st.button("🔄 Resume Extraction", type="primary"):
                    st.session_state["stop_requested"] = False
                    st.rerun()
            with col_resume2:
                st.info("Click 'Resume Extraction' to continue processing remaining chunks from where you left off.")
        else:
            st.success(f"✅ **Complete Results** - All chunks processed successfully. Found {len(extractions)} extractions.")
        
        # Build and display results table
        df = build_results_table(extractions, pages)
        st.dataframe(df, use_container_width=True)
        
        # Generate and display natural language summary
        st.subheader("📝 Natural Language Summary")
        with st.spinner("Generating summary..."):
            summary = generate_extraction_summary(provider, model, schema_prompt, extractions, int(timeout_s), int(max_tokens))
        
        # Display the summary in a nice format
        st.markdown("**AI-Generated Summary:**")
        st.markdown(summary)
        
        # Add a refresh button for the summary
        if st.button("🔄 Regenerate Summary", type="secondary"):
            with st.spinner("Regenerating summary..."):
                new_summary = generate_extraction_summary(provider, model, schema_prompt, extractions, int(timeout_s), int(max_tokens))
                st.markdown("**Updated AI-Generated Summary:**")
                st.markdown(new_summary)
        
        # Show extraction summary
        with st.expander("�� Extraction Summary", expanded=False):
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            
            with col_sum1:
                st.metric("Total Extractions", len(extractions))
            
            with col_sum2:
                unique_classes = len(set(item.get("extraction_class", "") for item in extractions))
                st.metric("Unique Classes", unique_classes)
            
            with col_sum3:
                total_pages = len(set(page for item in extractions for page in item.get("pages", [])))
                st.metric("Pages Covered", total_pages)
            
            # Show extraction class breakdown
            class_counts = {}
            for item in extractions:
                class_name = item.get("extraction_class", "Unknown")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if class_counts:
                st.markdown("**Extraction Class Breakdown:**")
                for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    st.markdown(f"- **{class_name}**: {count} items")

        # Show highlighted context
        st.subheader("🔍 Highlighted Context")
        page_to_spans = compute_highlight_spans(extractions, pages)
        render_highlighted_text(pages, page_to_spans)


if __name__ == "__main__":
    main()


