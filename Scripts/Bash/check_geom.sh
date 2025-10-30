 #!/usr/bin/env bash
shopt -s nullglob

# ─── CONFIG ─────────────────────────────────────────────────────────
DIR="/home/adunant/Documents/CEPH_PROJECTS/Firescape/Data/00_QGIS/SU/daily_gpkg"
files=( "$DIR"/*.gpkg )
total=${#files[@]}
bar_width=40

# ─── STATE ──────────────────────────────────────────────────────────
failures=()
i=0

echo "Checking $total files for real spatial geometry…"

for f in "${files[@]}"; do
  ((i++))
  name=$(basename "$f")
  date=${name%.gpkg}
  badmsg=""

  # 1) See if any geometry column is declared
  geom_entries=$(sqlite3 "$f" \
    "SELECT table_name||'|'||column_name FROM gpkg_geometry_columns;")

  if [[ -z "$geom_entries" ]]; then
    badmsg="no geometry columns declared"
  else
    # 2) For each (table|column), check for at least one non-NULL geometry
    has_good_geom=0
    IFS=$'\n'
    for entry in $geom_entries; do
      tbl=${entry%%|*}
      col=${entry#*|}
      # count non-null geometries
      cnt=$(sqlite3 "$f" \
        "SELECT COUNT(\"$col\") FROM \"$tbl\" WHERE \"$col\" IS NOT NULL;")
      if (( cnt > 0 )); then
        has_good_geom=1
        break
      fi
    done
    unset IFS

    if (( has_good_geom == 0 )); then
      badmsg="geometry column present but all NULL"
    fi
  fi

  # 3) Report any failures immediately
  if [[ -n "$badmsg" ]]; then
    printf "\n⚠️  %s.gpkg — %s\n" "$date" "$badmsg"
    failures+=( "$date" )
  fi

  # 4) Redraw the progress bar
  pct=$(( i * 100 / total ))
  filled=$(( pct * bar_width / 100 ))
  unfilled=$(( bar_width - filled ))
  bar="$(printf '%0.s█' $(seq 1 $filled))$(printf '%0.s░' $(seq 1 $unfilled))"
  printf "\r[%s] %3d%%  (%d/%d)" "$bar" "$pct" "$i" "$total"
done

echo -e "\n\nChecked $total files; ${#failures[@]} failures found."

if (( ${#failures[@]} )); then
  # Lexicographically smallest date is the oldest
  oldest=$(printf '%s\n' "${failures[@]}" | sort | head -n1)
  echo "Oldest failure date: $oldest"
else
  echo "All files contain valid, non-empty geometries."
fi
#!/usr/bin/env bash
shopt -s nullglob

# ─── CONFIG ─────────────────────────────────────────────────────────
DIR="/home/adunant/Documents/CEPH_PROJECTS/Firescape/Data/00_QGIS/SU/daily_gpkg"
files=( "$DIR"/*.gpkg )
total=${#files[@]}
bar_width=40

# ─── STATE ──────────────────────────────────────────────────────────
failures=()
i=0

echo "Checking $total files for real spatial geometry…"

for f in "${files[@]}"; do
  ((i++))
  name=$(basename "$f")
  date=${name%.gpkg}
  badmsg=""

  # 1) See if any geometry column is declared
  geom_entries=$(sqlite3 "$f" \
    "SELECT table_name||'|'||column_name FROM gpkg_geometry_columns;")

  if [[ -z "$geom_entries" ]]; then
    badmsg="no geometry columns declared"
  else
    # 2) For each (table|column), check for at least one non-NULL geometry
    has_good_geom=0
    IFS=$'\n'
    for entry in $geom_entries; do
      tbl=${entry%%|*}
      col=${entry#*|}
      # count non-null geometries
      cnt=$(sqlite3 "$f" \
        "SELECT COUNT(\"$col\") FROM \"$tbl\" WHERE \"$col\" IS NOT NULL;")
      if (( cnt > 0 )); then
        has_good_geom=1
        break
      fi
    done
    unset IFS

    if (( has_good_geom == 0 )); then
      badmsg="geometry column present but all NULL"
    fi
  fi

  # 3) Report any failures immediately
  if [[ -n "$badmsg" ]]; then
    printf "\n⚠️  %s.gpkg — %s\n" "$date" "$badmsg"
    failures+=( "$date" )
  fi

  # 4) Redraw the progress bar
  pct=$(( i * 100 / total ))
  filled=$(( pct * bar_width / 100 ))
  unfilled=$(( bar_width - filled ))
  bar="$(printf '%0.s█' $(seq 1 $filled))$(printf '%0.s░' $(seq 1 $unfilled))"
  printf "\r[%s] %3d%%  (%d/%d)" "$bar" "$pct" "$i" "$total"
done

echo -e "\n\nChecked $total files; ${#failures[@]} failures found."

if (( ${#failures[@]} )); then
  # Lexicographically smallest date is the oldest
  oldest=$(printf '%s\n' "${failures[@]}" | sort | head -n1)
  echo "Oldest failure date: $oldest"
else
  echo "All files contain valid, non-empty geometries."
fi
#!/usr/bin/env bash
shopt -s nullglob

# ─── CONFIG ─────────────────────────────────────────────────────────
DIR="/home/adunant/Documents/CEPH_PROJECTS/Firescape/Data/00_QGIS/SU/daily_gpkg"
files=( "$DIR"/*.gpkg )
total=${#files[@]}
bar_width=40

# ─── STATE ──────────────────────────────────────────────────────────
failures=()
i=0

echo "Checking $total files for real spatial geometry…"

for f in "${files[@]}"; do
  ((i++))
  name=$(basename "$f")
  date=${name%.gpkg}
  badmsg=""

  # 1) See if any geometry column is declared
  geom_entries=$(sqlite3 "$f" \
    "SELECT table_name||'|'||column_name FROM gpkg_geometry_columns;")

  if [[ -z "$geom_entries" ]]; then
    badmsg="no geometry columns declared"
  else
    # 2) For each (table|column), check for at least one non-NULL geometry
    has_good_geom=0
    IFS=$'\n'
    for entry in $geom_entries; do
      tbl=${entry%%|*}
      col=${entry#*|}
      # count non-null geometries
      cnt=$(sqlite3 "$f" \
        "SELECT COUNT(\"$col\") FROM \"$tbl\" WHERE \"$col\" IS NOT NULL;")
      if (( cnt > 0 )); then
        has_good_geom=1
        break
      fi
    done
    unset IFS

    if (( has_good_geom == 0 )); then
      badmsg="geometry column present but all NULL"
    fi
  fi

  # 3) Report any failures immediately
  if [[ -n "$badmsg" ]]; then
    printf "\n⚠️  %s.gpkg — %s\n" "$date" "$badmsg"
    failures+=( "$date" )
  fi

  # 4) Redraw the progress bar
  pct=$(( i * 100 / total ))
  filled=$(( pct * bar_width / 100 ))
  unfilled=$(( bar_width - filled ))
  bar="$(printf '%0.s█' $(seq 1 $filled))$(printf '%0.s░' $(seq 1 $unfilled))"
  printf "\r[%s] %3d%%  (%d/%d)" "$bar" "$pct" "$i" "$total"
done

echo -e "\n\nChecked $total files; ${#failures[@]} failures found."

if (( ${#failures[@]} )); then
  # Lexicographically smallest date is the oldest
  oldest=$(printf '%s\n' "${failures[@]}" | sort | head -n1)
  echo "Oldest failure date: $oldest"
else
  echo "All files contain valid, non-empty geometries."
fi
#!/usr/bin/env bash
shopt -s nullglob

# ─── CONFIG ─────────────────────────────────────────────────────────
DIR="/home/adunant/Documents/CEPH_PROJECTS/Firescape/Data/00_QGIS/SU/daily_gpkg"
files=( "$DIR"/*.gpkg )
total=${#files[@]}
bar_width=40

# ─── STATE ──────────────────────────────────────────────────────────
failures=()
i=0

echo "Checking $total files for real spatial geometry…"

for f in "${files[@]}"; do
  ((i++))
  name=$(basename "$f")
  date=${name%.gpkg}
  badmsg=""

  # 1) See if any geometry column is declared
  geom_entries=$(sqlite3 "$f" \
    "SELECT table_name||'|'||column_name FROM gpkg_geometry_columns;")

  if [[ -z "$geom_entries" ]]; then
    badmsg="no geometry columns declared"
  else
    # 2) For each (table|column), check for at least one non-NULL geometry
    has_good_geom=0
    IFS=$'\n'
    for entry in $geom_entries; do
      tbl=${entry%%|*}
      col=${entry#*|}
      # count non-null geometries
      cnt=$(sqlite3 "$f" \
        "SELECT COUNT(\"$col\") FROM \"$tbl\" WHERE \"$col\" IS NOT NULL;")
      if (( cnt > 0 )); then
        has_good_geom=1
        break
      fi
    done
    unset IFS

    if (( has_good_geom == 0 )); then
      badmsg="geometry column present but all NULL"
    fi
  fi

  # 3) Report any failures immediately
  if [[ -n "$badmsg" ]]; then
    printf "\n⚠️  %s.gpkg — %s\n" "$date" "$badmsg"
    failures+=( "$date" )
  fi

  # 4) Redraw the progress bar
  pct=$(( i * 100 / total ))
  filled=$(( pct * bar_width / 100 ))
  unfilled=$(( bar_width - filled ))
  bar="$(printf '%0.s█' $(seq 1 $filled))$(printf '%0.s░' $(seq 1 $unfilled))"
  printf "\r[%s] %3d%%  (%d/%d)" "$bar" "$pct" "$i" "$total"
done

echo -e "\n\nChecked $total files; ${#failures[@]} failures found."

if (( ${#failures[@]} )); then
  # Lexicographically smallest date is the oldest
  oldest=$(printf '%s\n' "${failures[@]}" | sort | head -n1)
  echo "Oldest failure date: $oldest"
else
  echo "All files contain valid, non-empty geometries."
fi
#!/usr/bin/env bash
shopt -s nullglob

# ─── CONFIG ─────────────────────────────────────────────────────────
DIR="/home/adunant/Documents/CEPH_PROJECTS/Firescape/Data/00_QGIS/SU/daily_gpkg"
files=( "$DIR"/*.gpkg )
total=${#files[@]}
bar_width=40

# ─── STATE ──────────────────────────────────────────────────────────
failures=()
i=0

echo "Checking $total files for real spatial geometry…"

for f in "${files[@]}"; do
  ((i++))
  name=$(basename "$f")
  date=${name%.gpkg}
  badmsg=""

  # 1) See if any geometry column is declared
  geom_entries=$(sqlite3 "$f" \
    "SELECT table_name||'|'||column_name FROM gpkg_geometry_columns;")

  if [[ -z "$geom_entries" ]]; then
    badmsg="no geometry columns declared"
  else
    # 2) For each (table|column), check for at least one non-NULL geometry
    has_good_geom=0
    IFS=$'\n'
    for entry in $geom_entries; do
      tbl=${entry%%|*}
      col=${entry#*|}
      # count non-null geometries
      cnt=$(sqlite3 "$f" \
        "SELECT COUNT(\"$col\") FROM \"$tbl\" WHERE \"$col\" IS NOT NULL;")
      if (( cnt > 0 )); then
        has_good_geom=1
        break
      fi
    done
    unset IFS

    if (( has_good_geom == 0 )); then
      badmsg="geometry column present but all NULL"
    fi
  fi

  # 3) Report any failures immediately
  if [[ -n "$badmsg" ]]; then
    printf "\n⚠️  %s.gpkg — %s\n" "$date" "$badmsg"
    failures+=( "$date" )
  fi

  # 4) Redraw the progress bar
  pct=$(( i * 100 / total ))
  filled=$(( pct * bar_width / 100 ))
  unfilled=$(( bar_width - filled ))
  bar="$(printf '%0.s█' $(seq 1 $filled))$(printf '%0.s░' $(seq 1 $unfilled))"
  printf "\r[%s] %3d%%  (%d/%d)" "$bar" "$pct" "$i" "$total"
done

echo -e "\n\nChecked $total files; ${#failures[@]} failures found."

if (( ${#failures[@]} )); then
  # Lexicographically smallest date is the oldest
  oldest=$(printf '%s\n' "${failures[@]}" | sort | head -n1)
  echo "Oldest failure date: $oldest"
else
  echo "All files contain valid, non-empty geometries."
fi
#!/usr/bin/env bash
shopt -s nullglob

# ─── CONFIG ─────────────────────────────────────────────────────────
DIR="/home/adunant/Documents/CEPH_PROJECTS/Firescape/Data/00_QGIS/SU/daily_gpkg"
files=( "$DIR"/*.gpkg )
total=${#files[@]}
bar_width=40

# ─── STATE ──────────────────────────────────────────────────────────
failures=()
i=0

echo "Checking $total files for real spatial geometry…"

for f in "${files[@]}"; do
  ((i++))
  name=$(basename "$f")
  date=${name%.gpkg}
  badmsg=""

  # 1) See if any geometry column is declared
  geom_entries=$(sqlite3 "$f" \
    "SELECT table_name||'|'||column_name FROM gpkg_geometry_columns;")

  if [[ -z "$geom_entries" ]]; then
    badmsg="no geometry columns declared"
  else
    # 2) For each (table|column), check for at least one non-NULL geometry
    has_good_geom=0
    IFS=$'\n'
    for entry in $geom_entries; do
      tbl=${entry%%|*}
      col=${entry#*|}
      # count non-null geometries
      cnt=$(sqlite3 "$f" \
        "SELECT COUNT(\"$col\") FROM \"$tbl\" WHERE \"$col\" IS NOT NULL;")
      if (( cnt > 0 )); then
        has_good_geom=1
        break
      fi
    done
    unset IFS

    if (( has_good_geom == 0 )); then
      badmsg="geometry column present but all NULL"
    fi
  fi

  # 3) Report any failures immediately
  if [[ -n "$badmsg" ]]; then
    printf "\n⚠️  %s.gpkg — %s\n" "$date" "$badmsg"
    failures+=( "$date" )
  fi

  # 4) Redraw the progress bar
  pct=$(( i * 100 / total ))
  filled=$(( pct * bar_width / 100 ))
  unfilled=$(( bar_width - filled ))
  bar="$(printf '%0.s█' $(seq 1 $filled))$(printf '%0.s░' $(seq 1 $unfilled))"
  printf "\r[%s] %3d%%  (%d/%d)" "$bar" "$pct" "$i" "$total"
done

echo -e "\n\nChecked $total files; ${#failures[@]} failures found."

if (( ${#failures[@]} )); then
  # Lexicographically smallest date is the oldest
  oldest=$(printf '%s\n' "${failures[@]}" | sort | head -n1)
  echo "Oldest failure date: $oldest"
else
  echo "All files contain valid, non-empty geometries."
fi
#!/usr/bin/env bash
shopt -s nullglob

# ─── CONFIG ─────────────────────────────────────────────────────────
DIR="/home/adunant/Documents/CEPH_PROJECTS/Firescape/Data/00_QGIS/SU/daily_gpkg"
files=( "$DIR"/*.gpkg )
total=${#files[@]}
bar_width=40

# ─── STATE ──────────────────────────────────────────────────────────
failures=()
i=0

echo "Checking $total files for real spatial geometry…"

for f in "${files[@]}"; do
  ((i++))
  name=$(basename "$f")
  date=${name%.gpkg}
  badmsg=""

  # 1) See if any geometry column is declared
  geom_entries=$(sqlite3 "$f" \
    "SELECT table_name||'|'||column_name FROM gpkg_geometry_columns;")

  if [[ -z "$geom_entries" ]]; then
    badmsg="no geometry columns declared"
  else
    # 2) For each (table|column), check for at least one non-NULL geometry
    has_good_geom=0
    IFS=$'\n'
    for entry in $geom_entries; do
      tbl=${entry%%|*}
      col=${entry#*|}
      # count non-null geometries
      cnt=$(sqlite3 "$f" \
        "SELECT COUNT(\"$col\") FROM \"$tbl\" WHERE \"$col\" IS NOT NULL;")
      if (( cnt > 0 )); then
        has_good_geom=1
        break
      fi
    done
    unset IFS

    if (( has_good_geom == 0 )); then
      badmsg="geometry column present but all NULL"
    fi
  fi

  # 3) Report any failures immediately
  if [[ -n "$badmsg" ]]; then
    printf "\n⚠️  %s.gpkg — %s\n" "$date" "$badmsg"
    failures+=( "$date" )
  fi

  # 4) Redraw the progress bar
  pct=$(( i * 100 / total ))
  filled=$(( pct * bar_width / 100 ))
  unfilled=$(( bar_width - filled ))
  bar="$(printf '%0.s█' $(seq 1 $filled))$(printf '%0.s░' $(seq 1 $unfilled))"
  printf "\r[%s] %3d%%  (%d/%d)" "$bar" "$pct" "$i" "$total"
done

echo -e "\n\nChecked $total files; ${#failures[@]} failures found."

if (( ${#failures[@]} )); then
  # Lexicographically smallest date is the oldest
  oldest=$(printf '%s\n' "${failures[@]}" | sort | head -n1)
  echo "Oldest failure date: $oldest"
else
  echo "All files contain valid, non-empty geometries."
fi
#!/usr/bin/env bash
shopt -s nullglob

# ─── CONFIG ─────────────────────────────────────────────────────────
DIR="/home/adunant/Documents/CEPH_PROJECTS/Firescape/Data/00_QGIS/SU/daily_gpkg"
files=( "$DIR"/*.gpkg )
total=${#files[@]}
bar_width=40

# ─── STATE ──────────────────────────────────────────────────────────
failures=()
i=0

echo "Checking $total files for real spatial geometry…"

for f in "${files[@]}"; do
  ((i++))
  name=$(basename "$f")
  date=${name%.gpkg}
  badmsg=""

  # 1) See if any geometry column is declared
  geom_entries=$(sqlite3 "$f" \
    "SELECT table_name||'|'||column_name FROM gpkg_geometry_columns;")

  if [[ -z "$geom_entries" ]]; then
    badmsg="no geometry columns declared"
  else
    # 2) For each (table|column), check for at least one non-NULL geometry
    has_good_geom=0
    IFS=$'\n'
    for entry in $geom_entries; do
      tbl=${entry%%|*}
      col=${entry#*|}
      # count non-null geometries
      cnt=$(sqlite3 "$f" \
        "SELECT COUNT(\"$col\") FROM \"$tbl\" WHERE \"$col\" IS NOT NULL;")
      if (( cnt > 0 )); then
        has_good_geom=1
        break
      fi
    done
    unset IFS

    if (( has_good_geom == 0 )); then
      badmsg="geometry column present but all NULL"
    fi
  fi

  # 3) Report any failures immediately
  if [[ -n "$badmsg" ]]; then
    printf "\n⚠️  %s.gpkg — %s\n" "$date" "$badmsg"
    failures+=( "$date" )
  fi

  # 4) Redraw the progress bar
  pct=$(( i * 100 / total ))
  filled=$(( pct * bar_width / 100 ))
  unfilled=$(( bar_width - filled ))
  bar="$(printf '%0.s█' $(seq 1 $filled))$(printf '%0.s░' $(seq 1 $unfilled))"
  printf "\r[%s] %3d%%  (%d/%d)" "$bar" "$pct" "$i" "$total"
done

echo -e "\n\nChecked $total files; ${#failures[@]} failures found."

if (( ${#failures[@]} )); then
  # Lexicographically smallest date is the oldest
  oldest=$(printf '%s\n' "${failures[@]}" | sort | head -n1)
  echo "Oldest failure date: $oldest"
else
  echo "All files contain valid, non-empty geometries."
fi
#!/usr/bin/env bash
shopt -s nullglob

# ─── CONFIG ─────────────────────────────────────────────────────────
DIR="/home/adunant/Documents/CEPH_PROJECTS/Firescape/Data/00_QGIS/SU/daily_gpkg"
files=( "$DIR"/*.gpkg )
total=${#files[@]}
bar_width=40

# ─── STATE ──────────────────────────────────────────────────────────
failures=()
i=0

echo "Checking $total files for real spatial geometry…"

for f in "${files[@]}"; do
  ((i++))
  name=$(basename "$f")
  date=${name%.gpkg}
  badmsg=""

  # 1) See if any geometry column is declared
  geom_entries=$(sqlite3 "$f" \
    "SELECT table_name||'|'||column_name FROM gpkg_geometry_columns;")

  if [[ -z "$geom_entries" ]]; then
    badmsg="no geometry columns declared"
  else
    # 2) For each (table|column), check for at least one non-NULL geometry
    has_good_geom=0
    IFS=$'\n'
    for entry in $geom_entries; do
      tbl=${entry%%|*}
      col=${entry#*|}
      # count non-null geometries
      cnt=$(sqlite3 "$f" \
        "SELECT COUNT(\"$col\") FROM \"$tbl\" WHERE \"$col\" IS NOT NULL;")
      if (( cnt > 0 )); then
        has_good_geom=1
        break
      fi
    done
    unset IFS

    if (( has_good_geom == 0 )); then
      badmsg="geometry column present but all NULL"
    fi
  fi

  # 3) Report any failures immediately
  if [[ -n "$badmsg" ]]; then
    printf "\n⚠️  %s.gpkg — %s\n" "$date" "$badmsg"
    failures+=( "$date" )
  fi

  # 4) Redraw the progress bar
  pct=$(( i * 100 / total ))
  filled=$(( pct * bar_width / 100 ))
  unfilled=$(( bar_width - filled ))
  bar="$(printf '%0.s█' $(seq 1 $filled))$(printf '%0.s░' $(seq 1 $unfilled))"
  printf "\r[%s] %3d%%  (%d/%d)" "$bar" "$pct" "$i" "$total"
done

echo -e "\n\nChecked $total files; ${#failures[@]} failures found."

if (( ${#failures[@]} )); then
  # Lexicographically smallest date is the oldest
  oldest=$(printf '%s\n' "${failures[@]}" | sort | head -n1)
  echo "Oldest failure date: $oldest"
else
  echo "All files contain valid, non-empty geometries."
fi
#!/usr/bin/env bash
shopt -s nullglob

# ─── CONFIG ─────────────────────────────────────────────────────────
DIR="/home/adunant/Documents/CEPH_PROJECTS/Firescape/Data/00_QGIS/SU/daily_gpkg"
files=( "$DIR"/*.gpkg )
total=${#files[@]}
bar_width=40

# ─── STATE ──────────────────────────────────────────────────────────
failures=()
i=0

echo "Checking $total files for real spatial geometry…"

for f in "${files[@]}"; do
  ((i++))
  name=$(basename "$f")
  date=${name%.gpkg}
  badmsg=""

  # 1) See if any geometry column is declared
  geom_entries=$(sqlite3 "$f" \
    "SELECT table_name||'|'||column_name FROM gpkg_geometry_columns;")

  if [[ -z "$geom_entries" ]]; then
    badmsg="no geometry columns declared"
  else
    # 2) For each (table|column), check for at least one non-NULL geometry
    has_good_geom=0
    IFS=$'\n'
    for entry in $geom_entries; do
      tbl=${entry%%|*}
      col=${entry#*|}
      # count non-null geometries
      cnt=$(sqlite3 "$f" \
        "SELECT COUNT(\"$col\") FROM \"$tbl\" WHERE \"$col\" IS NOT NULL;")
      if (( cnt > 0 )); then
        has_good_geom=1
        break
      fi
    done
    unset IFS

    if (( has_good_geom == 0 )); then
      badmsg="geometry column present but all NULL"
    fi
  fi

  # 3) Report any failures immediately
  if [[ -n "$badmsg" ]]; then
    printf "\n⚠️  %s.gpkg — %s\n" "$date" "$badmsg"
    failures+=( "$date" )
  fi

  # 4) Redraw the progress bar
  pct=$(( i * 100 / total ))
  filled=$(( pct * bar_width / 100 ))
  unfilled=$(( bar_width - filled ))
  bar="$(printf '%0.s█' $(seq 1 $filled))$(printf '%0.s░' $(seq 1 $unfilled))"
  printf "\r[%s] %3d%%  (%d/%d)" "$bar" "$pct" "$i" "$total"
done

echo -e "\n\nChecked $total files; ${#failures[@]} failures found."

if (( ${#failures[@]} )); then
  # Lexicographically smallest date is the oldest
  oldest=$(printf '%s\n' "${failures[@]}" | sort | head -n1)
  echo "Oldest failure date: $oldest"
else
  echo "All files contain valid, non-empty geometries."
fi
#!/usr/bin/env bash
shopt -s nullglob

# ─── CONFIG ─────────────────────────────────────────────────────────
DIR="/home/adunant/Documents/CEPH_PROJECTS/Firescape/Data/00_QGIS/SU/daily_gpkg"
files=( "$DIR"/*.gpkg )
total=${#files[@]}
bar_width=40

# ─── STATE ──────────────────────────────────────────────────────────
failures=()
i=0

echo "Checking $total files for real spatial geometry…"

for f in "${files[@]}"; do
  ((i++))
  name=$(basename "$f")
  date=${name%.gpkg}
  badmsg=""

  # 1) See if any geometry column is declared
  geom_entries=$(sqlite3 "$f" \
    "SELECT table_name||'|'||column_name FROM gpkg_geometry_columns;")

  if [[ -z "$geom_entries" ]]; then
    badmsg="no geometry columns declared"
  else
    # 2) For each (table|column), check for at least one non-NULL geometry
    has_good_geom=0
    IFS=$'\n'
    for entry in $geom_entries; do
      tbl=${entry%%|*}
      col=${entry#*|}
      # count non-null geometries
      cnt=$(sqlite3 "$f" \
        "SELECT COUNT(\"$col\") FROM \"$tbl\" WHERE \"$col\" IS NOT NULL;")
      if (( cnt > 0 )); then
        has_good_geom=1
        break
      fi
    done
    unset IFS

    if (( has_good_geom == 0 )); then
      badmsg="geometry column present but all NULL"
    fi
  fi

  # 3) Report any failures immediately
  if [[ -n "$badmsg" ]]; then
    printf "\n⚠️  %s.gpkg — %s\n" "$date" "$badmsg"
    failures+=( "$date" )
  fi

  # 4) Redraw the progress bar
  pct=$(( i * 100 / total ))
  filled=$(( pct * bar_width / 100 ))
  unfilled=$(( bar_width - filled ))
  bar="$(printf '%0.s█' $(seq 1 $filled))$(printf '%0.s░' $(seq 1 $unfilled))"
  printf "\r[%s] %3d%%  (%d/%d)" "$bar" "$pct" "$i" "$total"
done

echo -e "\n\nChecked $total files; ${#failures[@]} failures found."

if (( ${#failures[@]} )); then
  # Lexicographically smallest date is the oldest
  oldest=$(printf '%s\n' "${failures[@]}" | sort | head -n1)
  echo "Oldest failure date: $oldest"
else
  echo "All files contain valid, non-empty geometries."
fi
