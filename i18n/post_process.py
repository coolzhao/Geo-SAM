"""Post-process .ts files after pylupdate5 regeneration.

Approach: Process each <context> and <message> block individually.
1. Remove @default context
2. For each message, if it's unfinished, fill translation from lookup or obsolete
3. Remove obsolete messages after extracting their translations
"""

from __future__ import annotations

import re
import os

LANGUAGES = {
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish",
    "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
}

# All translations for every unique unfinished source text
T: dict[str, dict[str, str]] = {
    "Geo-SAM Settings": {
        "zh": "Geo-SAM 设置", "ja": "Geo-SAM 設定", "ko": "Geo-SAM 설정",
        "fr": "Paramètres Geo-SAM", "de": "Geo-SAM Einstellungen",
        "es": "Configuración de Geo-SAM", "pt": "Configurações do Geo-SAM",
        "ru": "Настройки Geo-SAM", "ar": "إعدادات Geo-SAM",
    },
    "{name} (Recommended)": {
        "zh": "{name}（推荐）", "ja": "{name}（推奨）", "ko": "{name}（권장）",
        "fr": "{name} (Recommandé)", "de": "{name} (Empfohlen)",
        "es": "{name} (Recomendado)", "pt": "{name} (Recomendado)",
        "ru": "{name} (рекомендуется)", "ar": "{name} (موصى به)",
    },
    "Custom Index URL:": {
        "zh": "自定义索引 URL：", "ja": "カスタムインデックス URL：", "ko": "사용자 정의 인덱스 URL:",
        "fr": "URL d'index personnalisé :", "de": "Benutzerdefinierte Index-URL:",
        "es": "URL de índice personalizado:", "pt": "URL de índice personalizado:",
        "ru": "Пользовательский URL индекса:", "ar": "URL الفهرس المخصص:",
    },
    "Clear Current Runtime Environment": {
        "zh": "清除当前运行环境", "ja": "現在のランタイム環境をクリア", "ko": "현재 런타임 환경 지우기",
        "fr": "Effacer l'environnement d'exécution actuel", "de": "Aktuelle Laufzeitumgebung löschen",
        "es": "Limpiar entorno de ejecución actual", "pt": "Limpar ambiente de execução atual",
        "ru": "Очистить текущую среду выполнения", "ar": "مسح بيئة التشغيل الحالية",
    },
    "Active Runtime: {profile}": {
        "zh": "当前运行环境：{profile}", "ja": "アクティブランタイム：{profile}", "ko": "활성 런타임: {profile}",
        "fr": "Runtime actif : {profile}", "de": "Aktive Laufzeit: {profile}",
        "es": "Tiempo de ejecución activo: {profile}", "pt": "Tempo de execução ativo: {profile}",
        "ru": "Активная среда выполнения: {profile}", "ar": "وقت التشغيل النشط: {profile}",
    },
    "Active Runtime: {active} | Selected Runtime: {selected}": {
        "zh": "当前运行环境：{active} | 已选运行环境：{selected}",
        "ja": "アクティブランタイム：{active} | 選択ランタイム：{selected}",
        "ko": "활성 런타임: {active} | 선택 런타임: {selected}",
        "fr": "Runtime actif : {active} | Runtime sélectionné : {selected}",
        "de": "Aktive Laufzeit: {active} | Ausgewählte Laufzeit: {selected}",
        "es": "Tiempo de ejecución activo: {active} | Tiempo de ejecución seleccionado: {selected}",
        "pt": "Tempo de execução ativo: {active} | Tempo de execução selecionado: {selected}",
        "ru": "Активная среда выполнения: {active} | Выбранная среда выполнения: {selected}",
        "ar": "وقت التشغيل النشط: {active} | وقت التشغيل المحدد: {selected}",
    },
    "Dependencies: {installed} installed, {missing} missing.": {
        "zh": "依赖项：已安装 {installed}，缺少 {missing}。",
        "ja": "依存関係：{installed} インストール済み、{missing} 未インストール。",
        "ko": "종속성: {installed} 설치됨, {missing} 누락.",
        "fr": "Dépendances : {installed} installée(s), {missing} manquante(s).",
        "de": "Abhängigkeiten: {installed} installiert, {missing} fehlen.",
        "es": "Dependencias: {installed} instaladas, {missing} faltantes.",
        "pt": "Dependências: {installed} instaladas, {missing} ausentes.",
        "ru": "Зависимости: {installed} установлено, {missing} отсутствует.",
        "ar": "التبعيات: {installed} مثبتة، {missing} مفقودة.",
    },
    "Dependencies: {installed} installed, {missing} missing — click 'Install Missing' to install.": {
        "zh": "依赖项：已安装 {installed}，缺失 {missing} — 点击「安装缺失」进行安装。",
        "ja": "依存関係：{installed} インストール済み、{missing} 未インストール — 「不足のインストール」をクリックしてください。",
        "ko": "종속성: {installed} 설치됨, {missing} 누락 — '누락 설치'를 클릭하여 설치하세요.",
        "fr": "Dépendances : {installed} installée(s), {missing} manquante(s) — cliquez sur « Installer manquantes ».",
        "de": "Abhängigkeiten: {installed} installiert, {missing} fehlen — klicken Sie auf 'Fehlende installieren'.",
        "es": "Dependencias: {installed} instaladas, {missing} faltantes — haga clic en 'Instalar faltantes'.",
        "pt": "Dependências: {installed} instaladas, {missing} ausentes — clique em 'Instalar Ausentes'.",
        "ru": "Зависимости: {installed} установлено, {missing} отсутствует — нажмите «Установить отсутствующие».",
        "ar": "التبعيات: {installed} مثبتة، {missing} مفقودة — انقر على 'تثبيت المفقودة'.",
    },
    "Already active.": {
        "zh": "当前已是激活状态。", "ja": "既にアクティブです。", "ko": "이미 활성화되어 있습니다.",
        "fr": "Déjà actif.", "de": "Bereits aktiv.",
        "es": "Ya está activo.", "pt": "Já está ativo.",
        "ru": "Уже активен.", "ar": "نشط بالفعل.",
    },
    "Install missing dependencies for '{selected}' first.": {
        "zh": "请先安装「{selected}」的缺失依赖项。",
        "ja": "まず「{selected}」の不足している依存関係をインストールしてください。",
        "ko": "먼저 '{selected}'의 누락된 종속성을 설치하세요.",
        "fr": "Installez d'abord les dépendances manquantes pour « {selected} ».",
        "de": "Installieren Sie zuerst die fehlenden Abhängigkeiten für '{selected}'.",
        "es": "Instale primero las dependencias faltantes para '{selected}'.",
        "pt": "Instale primeiro as dependências ausentes para '{selected}'.",
        "ru": "Сначала установите отсутствующие зависимости для «{selected}».",
        "ar": "قم بتثبيت التبعيات المفقودة لـ '{selected}' أولاً.",
    },
    "Torch Runtime": {
        "zh": "PyTorch 运行环境", "ja": "PyTorch ランタイム", "ko": "PyTorch 런타임",
        "fr": "Runtime PyTorch", "de": "PyTorch-Laufzeit",
        "es": "Tiempo de ejecución PyTorch", "pt": "Tempo de execução PyTorch",
        "ru": "Среда выполнения PyTorch", "ar": "وقت تشغيل PyTorch",
    },
    "Switch": {
        "zh": "切换", "ja": "切り替え", "ko": "전환",
        "fr": "Changer", "de": "Wechseln",
        "es": "Cambiar", "pt": "Trocar",
        "ru": "Переключить", "ar": "تبديل",
    },
    "Runtime:": {
        "zh": "运行环境：", "ja": "ランタイム：", "ko": "런타임:",
        "fr": "Runtime :", "de": "Laufzeit:",
        "es": "Tiempo de ejecución:", "pt": "Tempo de execução:",
        "ru": "Среда выполнения:", "ar": "وقت التشغيل:",
    },
    "Selected runtime '{selected}': {installed} installed, {missing} missing.": {
        "zh": "所选运行环境「{selected}」：已安装 {installed}，缺失 {missing}。",
        "ja": "選択したランタイム「{selected}」：{installed} インストール済み、{missing} 未インストール。",
        "ko": "선택한 런타임 '{selected}': {installed} 설치됨, {missing} 누락.",
        "fr": "Runtime sélectionné « {selected} » : {installed} installée(s), {missing} manquante(s).",
        "de": "Ausgewählte Laufzeit '{selected}': {installed} installiert, {missing} fehlend.",
        "es": "Tiempo de ejecución seleccionado '{selected}': {installed} instaladas, {missing} faltantes.",
        "pt": "Tempo de execução selecionado '{selected}': {installed} instaladas, {missing} ausentes.",
        "ru": "Выбранная среда выполнения «{selected}»: {installed} установлено, {missing} отсутствует.",
        "ar": "وقت التشغيل المحدد '{selected}': {installed} مثبتة، {missing} مفقودة.",
    },
    "Target runtime: {profile}": {
        "zh": "目标运行环境：{profile}", "ja": "ターゲットランタイム：{profile}", "ko": "대상 런타임: {profile}",
        "fr": "Runtime cible : {profile}", "de": "Ziel-Laufzeit: {profile}",
        "es": "Tiempo de ejecución de destino: {profile}", "pt": "Tempo de execução de destino: {profile}",
        "ru": "Целевая среда выполнения: {profile}", "ar": "وقت التشغيل الهدف: {profile}",
    },
    "Switch Torch Runtime": {
        "zh": "切换 PyTorch 运行环境", "ja": "PyTorch ランタイムを切り替え", "ko": "PyTorch 런타임 전환",
        "fr": "Changer de runtime PyTorch", "de": "PyTorch-Laufzeit wechseln",
        "es": "Cambiar tiempo de ejecución de PyTorch", "pt": "Trocar tempo de execução do PyTorch",
        "ru": "Переключить среду выполнения PyTorch", "ar": "تبديل وقت تشغيل PyTorch",
    },
    "Switching the torch runtime will unload loaded models.\n\nTarget runtime: {profile}\n\nContinue?": {
        "zh": "切换 PyTorch 运行环境将卸载已加载的模型。\n\n目标运行环境：{profile}\n\n是否继续？",
        "ja": "PyTorch ランタイムを切り替えると、ロード済みモデルがアンロードされます。\n\nターゲットランタイム：{profile}\n\n続行しますか？",
        "ko": "PyTorch 런타임을 전환하면 로드된 모델이 언로드됩니다.\n\n대상 런타임: {profile}\n\n계속하시겠습니까?",
        "fr": "Le changement de runtime PyTorch déchargera les modèles chargés.\n\nRuntime cible : {profile}\n\nContinuer ?",
        "de": "Durch Wechsel der PyTorch-Laufzeit werden geladene Modelle entladen.\n\nZiel-Laufzeit: {profile}\n\nFortfahren?",
        "es": "Cambiar el tiempo de ejecución de PyTorch descargará los modelos cargados.\n\nTiempo de ejecución de destino: {profile}\n\n¿Continuar?",
        "pt": "Trocar o tempo de execução do PyTorch descarregará os modelos carregados.\n\nTempo de execução de destino: {profile}\n\nContinuar?",
        "ru": "Переключение среды выполнения PyTorch выгрузит загруженные модели.\n\nЦелевая среда выполнения: {profile}\n\nПродолжить?",
        "ar": "سيؤدي تبديل وقت تشغيل PyTorch إلى إلغاء تحميل النماذج المحملة.\n\nوقت التشغيل الهدف: {profile}\n\nهل تريد المتابعة؟",
    },
    "{error}\n\nUse 'Install Missing' to install dependencies for the '{profile}' runtime first.": {
        "zh": "{error}\n\n请先使用「安装缺失」为「{profile}」运行环境安装依赖项。",
        "ja": "{error}\n\nまず「不足のインストール」を使用して、「{profile}」ランタイムの依存関係をインストールしてください。",
        "ko": "{error}\n\n먼저 '누락 설치'를 사용하여 '{profile}' 런타임의 종속성을 설치하세요.",
        "fr": "{error}\n\nUtilisez d'abord « Installer manquantes » pour installer les dépendances du runtime « {profile} ».",
        "de": "{error}\n\nVerwenden Sie zuerst 'Fehlende installieren', um Abhängigkeiten für die Laufzeit '{profile}' zu installieren.",
        "es": "{error}\n\nPrimero use 'Instalar faltantes' para instalar las dependencias del tiempo de ejecución '{profile}'.",
        "pt": "{error}\n\nUse 'Instalar Ausentes' primeiro para instalar as dependências do tempo de execução '{profile}'.",
        "ru": "{error}\n\nСначала используйте «Установить отсутствующие» для установки зависимостей среды выполнения «{profile}».",
        "ar": "{error}\n\nاستخدم 'تثبيت المفقودة' أولاً لتثبيت تبعيات وقت التشغيل '{profile}'.",
    },
    "Switch failed: {error}": {
        "zh": "切换失败：{error}", "ja": "切り替えに失敗しました：{error}", "ko": "전환 실패: {error}",
        "fr": "Échec du changement : {error}", "de": "Wechsel fehlgeschlagen: {error}",
        "es": "Error al cambiar: {error}", "pt": "Falha ao alternar: {error}",
        "ru": "Ошибка переключения: {error}", "ar": "فشل التبديل: {error}",
    },
    "Switched to {profile}.": {
        "zh": "已切换到 {profile}。", "ja": "{profile} に切り替えました。", "ko": "{profile}로 전환되었습니다.",
        "fr": "Basculement vers {profile} effectué.", "de": "Zu {profile} gewechselt.",
        "es": "Cambiado a {profile}.", "pt": "Alternado para {profile}.",
        "ru": "Переключено на {profile}.", "ar": "تم التبديل إلى {profile}.",
    },
    "Copied install commands to clipboard.": {
        "zh": "已将安装命令复制到剪贴板。", "ja": "インストールコマンドをクリップボードにコピーしました。",
        "ko": "설치 명령이 클립보드에 복사되었습니다.",
        "fr": "Commandes d'installation copiées dans le presse-papiers.",
        "de": "Installationsbefehle in die Zwischenablage kopiert.",
        "es": "Comandos de instalación copiados al portapapeles.",
        "pt": "Comandos de instalação copiados para a área de transferência.",
        "ru": "Команды установки скопированы в буфер обмена.",
        "ar": "تم نسخ أوامر التثبيت إلى الحافظة.",
    },
    "Failed to build command: {error}": {
        "zh": "构建命令失败：{error}", "ja": "コマンドの構築に失敗しました：{error}", "ko": "명령 빌드 실패: {error}",
        "fr": "Échec de la construction de la commande : {error}",
        "de": "Befehlserstellung fehlgeschlagen: {error}",
        "es": "Error al construir el comando: {error}", "pt": "Falha ao construir comando: {error}",
        "ru": "Не удалось собрать команду: {error}", "ar": "فشل بناء الأمر: {error}",
    },
    "\n\nOn Windows, files may be locked by running processes. Please restart QGIS and try again.": {
        "zh": "\n\n在 Windows 上，文件可能被正在运行的进程锁定。请重启 QGIS 后重试。",
        "ja": "\n\nWindows では、実行中のプロセスによってファイルがロックされている可能性があります。QGIS を再起動してからもう一度お試しください。",
        "ko": "\n\nWindows에서는 실행 중인 프로세스에 의해 파일이 잠겨 있을 수 있습니다. QGIS를 다시 시작한 후 다시 시도하세요.",
        "fr": "\n\nSous Windows, les fichiers peuvent être verrouillés par des processus en cours. Veuillez redémarrer QGIS et réessayer.",
        "de": "\n\nUnter Windows können Dateien durch laufende Prozesse gesperrt sein. Bitte starten Sie QGIS neu und versuchen Sie es erneut.",
        "es": "\n\nEn Windows, los archivos pueden estar bloqueados por procesos en ejecución. Reinicie QGIS e intente de nuevo.",
        "pt": "\n\nNo Windows, arquivos podem estar bloqueados por processos em execução. Reinicie o QGIS e tente novamente.",
        "ru": "\n\nВ Windows файлы могут быть заблокированы запущенными процессами. Перезапустите QGIS и повторите попытку.",
        "ar": "\n\nعلى Windows، قد تكون الملفات مقفلة بواسطة عمليات قيد التشغيل. يرجى إعادة تشغيل QGIS والمحاولة مرة أخرى.",
    },
    "Plugin-managed dependencies: {current_size} in current runtime, {total_size} across {folder_count} runtime folder(s). Current path: {current_path}": {
        "zh": "插件依赖目录：当前运行环境 {current_size}，{folder_count} 个运行环境文件夹共 {total_size}。当前路径：{current_path}",
        "ja": "プラグイン依存ディレクトリ：現在の実行環境 {current_size}、{folder_count} 個の実行環境フォルダで {total_size}。現在のパス：{current_path}",
        "ko": "플러그인 종속 디렉토리: 현재 실행 환경 {current_size}, {folder_count}개 실행 환경 폴더 총 {total_size}. 현재 경로: {current_path}",
        "fr": "Dépendances du plugin : {current_size} dans l'environnement actuel, {total_size} dans {folder_count} dossier(s) d'environnement. Chemin actuel : {current_path}",
        "de": "Plugin-Abhängigkeiten: {current_size} in aktueller Umgebung, {total_size} in {folder_count} Umgebungsordner(n). Aktueller Pfad: {current_path}",
        "es": "Dependencias del plugin: {current_size} en el entorno actual, {total_size} en {folder_count} carpeta(s) de entorno. Ruta actual: {current_path}",
        "pt": "Dependências do plugin: {current_size} no ambiente atual, {total_size} em {folder_count} pasta(s) de ambiente. Caminho atual: {current_path}",
        "ru": "Зависимости плагина: {current_size} в текущей среде, {total_size} в {folder_count} папке(ах) среды. Текущий путь: {current_path}",
        "ar": "تبعيات الإضافة: {current_size} في البيئة الحالية، {total_size} في {folder_count} مجلد(ات) بيئة. المسار الحالي: {current_path}",
    },
}


def xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("'", "&apos;")


def xml_unescape(text: str) -> str:
    return text.replace("&apos;", "'").replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")


def process_file(filepath: str, lang: str) -> None:
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if we're entering @default context
        if "<name>@default</name>" in line:
            # Remove the preceding <context> line too
            if result and "<context>" in result[-1]:
                result.pop()
            # Skip until </context>
            i += 1
            while i < len(lines) and "</context>" not in lines[i]:
                i += 1
            i += 1  # Skip </context>
            continue

        # Check if we're entering a <message> block
        if "<message>" in line.strip():
            # Collect the entire message block
            msg_lines: list[str] = []
            msg_lines.append(line)
            i += 1
            while i < len(lines) and "</message>" not in lines[i]:
                msg_lines.append(lines[i])
                i += 1
            if i < len(lines):
                msg_lines.append(lines[i])  # Include </message>
                i += 1

            msg_text = "".join(msg_lines)

            # Determine message type
            if 'type="obsolete"' in msg_text:
                # Skip obsolete messages entirely
                continue
            elif 'type="unfinished"' in msg_text:
                # Try to fill the translation
                # Extract source text
                src_match = re.search(r"<source>(.*?)</source>", msg_text, re.DOTALL)
                if src_match:
                    src_escaped = src_match.group(1)
                    src_decoded = xml_unescape(src_escaped)

                    # Look up translation
                    trans = None
                    # Exact key match (XML-escaped)
                    if src_decoded in T and lang in T[src_decoded]:
                        trans = T[src_decoded][lang]
                    # Try with &apos; 
                    elif src_escaped in T and lang in T[src_escaped]:
                        trans = T[src_escaped][lang]

                    if trans is not None:
                        trans_escaped = xml_escape(trans)
                        # Replace unfinished translation (may have old content)
                        msg_text = re.sub(
                            r'<translation type="unfinished">.*?</translation>',
                            f"<translation>{trans_escaped}</translation>",
                            msg_text,
                            flags=re.DOTALL,
                        )

                result.append(msg_text)
            else:
                # Keep as-is
                result.append(msg_text)
        else:
            result.append(line)
            i += 1

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(result)


def main() -> None:
    i18n_dir = os.path.dirname(os.path.abspath(__file__))

    for lang_code in LANGUAGES:
        filename = f"GeoSAM_{lang_code}.ts"
        filepath = os.path.join(i18n_dir, filename)
        if not os.path.exists(filepath):
            print(f"  SKIP: {filepath} not found")
            continue

        print(f"Processing {lang_code} ({LANGUAGES[lang_code]})...", end=" ")
        process_file(filepath, lang_code)
        print("OK")

    # Stats
    print("\n=== Final Stats ===")
    all_ok = True
    for lang_code in LANGUAGES:
        filepath = os.path.join(i18n_dir, f"GeoSAM_{lang_code}.ts")
        if not os.path.exists(filepath):
            continue
        with open(filepath, encoding="utf-8") as f:
            c = f.read()
        lines = c.count("\n") + 1
        obs = c.count('type="obsolete"')
        unf = c.count('type="unfinished"')
        default = c.count("@default")
        status = "OK" if (obs == 0 and unf == 0 and default == 0) else "ISSUES"
        if status != "OK":
            all_ok = False
        print(f"  {lang_code}: {lines} lines | obsolete={obs} unfinished={unf} @default={default} | {status}")

    if all_ok:
        print("\nAll files clean!")
    else:
        print("\nSome issues remain.")


if __name__ == "__main__":
    main()
