; 竞彩足球预测系统 - Inno Setup 安装脚本
; 用法: ISCC.exe install.iss

#define MyAppName "竞彩足球预测"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "FootballPredictor"
#define MyAppExeName "FootballPredictor.exe"

[Setup]
AppId={{8E3C1A5B-2F4D-4C6E-9A7B-1D2E3F4A5B6C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\FootballPredictor
DefaultGroupName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
OutputDir=installer_output
OutputBaseFilename=竞彩足球预测_Setup
SetupIconFile=app_icon.ico
; 中文安装界面
LanguageDetectionMethod=none
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"; Flags: checkedonce

[Files]
; 打包整个 FootballPredictor 目录（dist_pkg_win4 = 窗口模式无黑框，含置信度标注）
Source: "dist_pkg_win4\FootballPredictor\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "立即运行 {#MyAppName}"; Flags: nowait postinstall skipifsilent
