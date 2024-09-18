## 環境構築に使ってコマンドリスト
# Command History

1. `brew -v`
2. `python -v`
3. `whitch python`
4. `which python`
5. `pyenv --version`
6. `python3 -m venv pytorch_env`
7. `pip3 install --upgrade pip`
8. `source venv/bin/activate`
9. `python3 -m venv venv`
10. `source venv/bin/activate`
11. `source venv/bin/activate`
12. `pip install --upgrade pip`
13. `pip install torch torchvision torchaudio`
14. `python -c "import torch; print(torch.__version__)"` // pytorchのversionを確認する
15. `python3 version`
16. `python3 --version`

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

`venv` は以下の役目

1. 独立した Python 環境を作成します。
2. その環境専用の Python バイナリをコピーまたはシンボリックリンクします。
3. 環境をアクティブ化するためのスクリプトを作成します。
4. （オプションで）基本的なパッケージ（pip など）をインストールします。

要は仮想環境を作って、pythonの実行環境を作ってくれる、Google Colabなどを使わなくてもいいし、他のPythonのプロジェクトのversionの衝突を防ぐことができる

この仮想環境を使用することで、プロジェクト固有の依存関係を他のプロジェクトやシステム全体の Python 環境から分離できます。これにより、異なるプロジェクト間での依存関係の衝突を防ぎ、クリーンで再現可能な開発環境を維持することができます。

仮想環境を使用することは、特に複数のプロジェクトを扱う場合や、特定のバージョンの依存関係が必要な場合に非常に有用です。

## 開発を始めるとき

仮想環境の使用を終了する場合は、以下のコマンドを使用します：
deactivate
また、プロジェクトで作業を再開する際は、仮想環境を再度アクティブにする必要があります：
source venv/bin/activate