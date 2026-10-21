import csv
import hashlib
import os
import re
import glob


class JavaCommentExtractor:

  def __init__(self, repo_dir="./spring_framework_repo"):
    self.repo_dir = repo_dir

  def _extract_comments_from_content(self, content):
    """Parses Java source code content to extract comments along with line numbers using regex/state scanning."""
    comments = []
    # Regular expression to match single-line (//) and block/javadoc comments (/* ... */)
    pattern = re.compile(
        r'(?P<single>//.*?$)|(?P<block>/\*.*?\*/)',
        re.DOTALL | re.MULTILINE
    )

    for match in pattern.finditer(content):
      comment_raw = match.group('single') or match.group('block')
      
      # Calculate line number by counting newlines up to the match start index
      line_start = content[:match.start()].count('\n') + 1
      line_end = line_start + comment_raw.count('\n')

      # Clean the comment text (strip markers //, /*, */, and leading/trailing asterisks *)
      cleaned_text = self._clean_comment_text(comment_raw)

      # Filter out empty or trivial comments (e.g., license headers, empty lines)
      if len(cleaned_text) > 3 and not cleaned_text.startswith('Copyright'):
        comments.append({
            'line_start': line_start,
            'line_end': line_end,
            'comment_text': cleaned_text,
            'comment_hash': hashlib.md5(cleaned_text.encode('utf-8')).hexdigest()
        })
    return comments

  def _clean_comment_text(self, raw_text):
    """Strips comment syntax markers to isolate the actual developer text."""
    # Remove block comment markers
    text = re.sub(r'^/\*\*?|\*/$', '', raw_text.strip())
    # Remove leading asterisks on lines within block comments
    lines = [re.sub(r'^\s*\*\s?', '', line) for line in text.split('\n')]
    # Remove single-line comment markers
    lines = [re.sub(r'^\s*//\s?', '', line) for line in lines]
    return ' '.join([l.strip() for l in lines if l.strip()])

  def extract_version_comments(self, tag_name, output_csv_path):
    """Walks through the repository, extracts comments from all .java files, and saves to CSV."""
    print(f"Extracting comments for {tag_name}...")
    
    all_comments = []
    java_files = glob.glob(os.path.join(self.repo_dir, '**', '*.java'), recursive=True)

    for file_path in java_files:
      rel_path = os.path.relpath(file_path, self.repo_dir)
      
      # Skip test folders or build directories if you want to focus strictly on core source code, 
      # or keep them if you want comprehensive analysis. (Let's keep src/main for core production debt).
      if 'src' not in rel_path:
        continue

      try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
          content = f.read()
        
        file_comments = self._extract_comments_from_content(content)
        for fc in file_comments:
          fc['file_path'] = rel_path
          all_comments.append(fc)
      except Exception as e:
        # Ignore individual file read issues
        continue

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
      writer = csv.DictWriter(
          f, fieldnames=['file_path', 'line_start', 'line_end', 'comment_text', 'comment_hash']
      )
      writer.writeheader()
      writer.writerows(all_comments)

    print(f"Saved {len(all_comments)} comments for {tag_name} to {output_csv_path}")
    return output_csv_path


# --- Quick Test Snippet ---
if __name__ == '__main__':
  from repo_manager import SpringRepoManager
  manager = SpringRepoManager()
  tags = manager.get_v53_tags()
  
  # Test extraction on just the first patch (v5.3.0) to verify
  manager.checkout_version(tags[0])
  extractor = JavaCommentExtractor()
  extractor.extract_version_comments(tags[0], f'./data/raw_comments/{tags[0]}.csv')