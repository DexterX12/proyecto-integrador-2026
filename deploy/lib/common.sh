# shellcheck shell=bash
# Funciones comunes de deploy.sh, destroy.sh y setup_wsl.sh (se cargan con `source`, no se ejecutan).

# aws (v2) y osls se instalan en ~/.local/bin por setup_wsl.sh (sin sudo); se añade al PATH solo si falta
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:$PATH" ;; esac
export AWS_PAGER=""                         # sin paginador interactivo en los comandos de aws
DRY_RUN="${DRY_RUN:-0}"

if [ -t 1 ]; then C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_DIM=""; C_OFF=""; fi

info() { printf '%s%s%s\n' "$C_CYAN" "$*" "$C_OFF"; }
ok()   { printf '  %s%s%s\n' "$C_GREEN" "$*" "$C_OFF"; }
warn() { printf '  %sAVISO: %s%s\n' "$C_YELLOW" "$*" "$C_OFF" >&2; }
fail() { printf '\n%sERROR: %s%s\n' "$C_RED" "$*" "$C_OFF" >&2; exit 1; }
dry()  { [ "$DRY_RUN" = "1" ]; }

is_true() { case "${1:-}" in 1|true|TRUE|True|yes|YES|si|SI|y|Y) return 0 ;; *) return 1 ;; esac; }

# load_env <archivo>: carga CLAVE=valor SIN ejecutar el archivo (no es `source`).
# Ignora comentarios y líneas vacías, acepta comillas y finales de línea de Windows (CRLF).
# Lo que ya esté definido en el entorno de la shell tiene prioridad sobre el archivo.
load_env() {
  local file="$1" line key val
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    if [[ ! "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      warn "línea ignorada en $file (formato CLAVE=valor): ${line:0:25}"; continue
    fi
    key="${BASH_REMATCH[2]}"; val="${BASH_REMATCH[3]}"
    if   [[ "$val" =~ ^\"(.*)\"[[:space:]]*(#.*)?$ ]]; then val="${BASH_REMATCH[1]}"
    elif [[ "$val" =~ ^\'(.*)\'[[:space:]]*(#.*)?$ ]]; then val="${BASH_REMATCH[1]}"
    else val="${val%%[[:space:]]#*}"; val="${val%"${val##*[![:space:]]}"}"; fi
    [ -n "$val" ] || continue                      # CLAVE= vacía equivale a no definirla (AWS CLI no tolera AWS_PROFILE vacío)
    if [ -z "${!key+x}" ]; then export "$key=$val"; fi
  done < "$file"
}

# q <args de aws>: consulta de solo lectura (se ejecuta también en --dry-run). Falla en silencio.
q() { aws "$@" 2>/dev/null; }

# act <args de aws>: comando que modifica algo; en --dry-run solo se imprime.
act() {
  if dry; then printf '  %s[dry-run] aws %s%s\n' "$C_DIM" "$*" "$C_OFF"; return 0; fi
  aws "$@" || fail "falló 'aws ${*:1:3}'"
}
act_soft() {
  if dry; then printf '  %s[dry-run] aws %s%s\n' "$C_DIM" "$*" "$C_OFF"; return 0; fi
  aws "$@" || { warn "falló 'aws ${*:1:3}', se continúa"; return 0; }
}

stack_output() { # stack_output <stack> <OutputKey>  -> imprime el valor o nada
  local v
  v="$(q cloudformation describe-stacks --stack-name "$1" \
        --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text)" || v=""
  [ "$v" = "None" ] && v=""
  printf '%s' "$v"
}

python_bin() { # intérprete con boto3/pyarrow: el venv de setup_wsl.sh, o python3 del sistema
  if [ -n "${FASHREC_PYTHON:-}" ]; then printf '%s' "$FASHREC_PYTHON"
  elif [ -x "$HOME/.venvs/fashrec/bin/python" ]; then printf '%s' "$HOME/.venvs/fashrec/bin/python"
  else printf 'python3'; fi
}
