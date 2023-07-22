"""
実行ファイル
学習もテストもここから行う
"""
import argparse


if __name__ == "__main__":
    # コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='test mode')