# ファイルを C++ のバイト配列にして埋め込むための生成スクリプト（cmake -P で呼ぶ）
#   cmake -DINPUT=<入力> -DOUTPUT=<出力 .cpp> -DSYMBOL=<配列名> -P embed_file.cmake
# 配列 <SYMBOL>（末尾に終端の 0 を 1 つ足す）と、元のサイズ <SYMBOL>_size を extern "C" で定義する。
file(READ "${INPUT}" content HEX)
string(LENGTH "${content}" hex_length)
math(EXPR size "${hex_length} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${content}")
file(WRITE "${OUTPUT}"
"// ${INPUT} から生成したファイル（編集しない）
extern \"C\" {
extern const unsigned char ${SYMBOL}[];
extern const unsigned long ${SYMBOL}_size;
const unsigned char ${SYMBOL}[] = {${bytes}0x00};
const unsigned long ${SYMBOL}_size = ${size}UL;
}
")
