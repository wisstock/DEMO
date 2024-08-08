=========
SOMETHING
=========

git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/ceo.jpg' \
  --prune-empty --tag-name-filter cat -- --all