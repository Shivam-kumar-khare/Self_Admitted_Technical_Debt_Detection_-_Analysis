import os
import git


class SpringRepoManager:

  def __init__(
      self, repo_url="https://github.com/spring-projects/spring-framework.git",
      repo_dir="./spring_framework_repo"
  ):
    self.repo_url = repo_url
    self.repo_dir = repo_dir
    self.repo = None
    self._initialize_repo()

  def _initialize_repo(self):
    """Clones the repository if it doesn't exist, otherwise loads it."""
    if not os.path.exists(self.repo_dir):
      print(
          f"Cloning Spring Framework into {self.repo_dir}... (this may take a"
          " few minutes)"
      )
      self.repo = git.Repo.clone_from(self.repo_url, self.repo_dir)
      print("Cloning complete.")
    else:
      print(f"Loading existing repository from {self.repo_dir}...")
      self.repo = git.Repo(self.repo_dir)

  def get_v53_tags(self):
    """Retrieves and numerically sorts all v5.3.x release tags."""
    # Filter tags starting with 'v5.3.' and ignore release candidates/milestones if desired
    tags = [
        tag.name
        for tag in self.repo.tags
        if tag.name.startswith("v5.3.")
        and "-" not in tag.name  # Keeps only stable releases (e.g., v5.3.39)
    ]

    # Sort numerically by patch version number (v5.3.0, v5.3.1, ..., v5.3.39)
    sorted_tags = sorted(tags, key=lambda x: int(x.split(".")[2]))
    return sorted_tags

  def checkout_version(self, tag_name):
    """Checks out a specific version tag in a detached HEAD state."""
    print(f"Checking out version: {tag_name}")
    self.repo.git.checkout(tag_name)
    return self.repo_dir


# --- Quick Test Snippet ---
if __name__ == "__main__":
  manager = SpringRepoManager()
  v53_tags = manager.get_v53_tags()
  print(f"Total discovered v5.3 stable patches: {len(v53_tags)}")
  print(f"First 5 patches: {v53_tags[:5]}")
  print(f"Last 5 patches: {v53_tags[-5:]}")