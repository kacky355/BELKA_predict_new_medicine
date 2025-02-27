#!/bin/bash

# mambaを初期化
mamba init -q

# Pythonを最新にアップデート
mamba update python -qy

# Jupyter Notebook のインストール
mamba install jupyter -y

# 追加パッケージのインストール
# コメントアウトを外し、<install packages> の部分を置き換えてください
# mamba install -qy <install packages>  # パッケージのインストール

# 全パッケージのアップデート
mamba update -qy --all

# キャッシュ等を削除
mamba clean -qafy
