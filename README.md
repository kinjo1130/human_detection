## 環境構築に使ってコマンドリスト
 history
 1082  brew -v
 1083  python -v
 1084  whitch python
 1085  which python
 1086  pyenv --version
 1087  python3 -m venv pytorch_env
 1088  pip3 install --upgrade pip
 1089  source venv/bin/activate
 1090  python3 -m venv venv
 1091  source venv/bin/activate\n\n
 1092  source venv/bin/activate
 1093  pip install --upgrade pip
 1094  pip install torch torchvision torchaudio
 1095  python -c "import torch; print(torch.__version__)" // pytorchのversionを確認する
 1096  python3 version
 1097  python3 --version

## 環境
- pytorch version: 2.4.1
- python version: 3.11.6
- venv: pythonの仮想環境

`python3 -m venv venv` というコマンドについて説明します。

このコマンドは Python の標準ライブラリに含まれる `venv` モジュールを使用して、新しい仮想環境を作成しています。

詳細な説明：

1. `python3`: Python 3 のインタープリタを呼び出しています。

2. `-m venv`: `-m` フラグは、`venv` という名前のモジュールを実行するよう Python に指示しています。`venv` は Python の標準ライブラリに含まれる仮想環境作成用のモジュールです。

3. 最後の `venv`: これは作成する仮想環境の名前とディレクトリ名を指定しています。この場合、現在のディレクトリに `venv` という名前のディレクトリが作成され、その中に仮想環境がセットアップされます。

使用しているツール：
- Python 3 の標準ライブラリに含まれる `venv` モジュール

`venv` は以下の役割を果たします：

1. 独立した Python 環境を作成します。
2. その環境専用の Python バイナリをコピーまたはシンボリックリンクします。
3. 環境をアクティブ化するためのスクリプトを作成します。
4. （オプションで）基本的なパッケージ（pip など）をインストールします。

この仮想環境を使用することで、プロジェクト固有の依存関係を他のプロジェクトやシステム全体の Python 環境から分離できます。これにより、異なるプロジェクト間での依存関係の衝突を防ぎ、クリーンで再現可能な開発環境を維持することができます。

仮想環境を使用することは、特に複数のプロジェクトを扱う場合や、特定のバージョンの依存関係が必要な場合に非常に有用です。

## 開発を始めるとき

仮想環境の使用を終了する場合は、以下のコマンドを使用します：
deactivate
また、プロジェクトで作業を再開する際は、仮想環境を再度アクティブにする必要があります：
source venv/bin/activate