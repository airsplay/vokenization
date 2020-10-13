# Copyright (c) 2019-present, Facebook, Inc.
# Copy frrom https://github.com/facebookresearch/XLM
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# Usage: ./get-data-wiki.sh $lg (en)
#

set -e

lg=$1  # input language

# data path
WIKI_PATH=data/wiki-cased-untokenized
MAIN_PATH=$WIKI_PATH

# tools paths
TOOLS_PATH=$MAIN_PATH/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
REMOVE_ACCENT=$TOOLS_PATH/remove_accent.py

# Wiki data
WIKI_DUMP_NAME=${lg}wiki-latest-pages-articles.xml.bz2
WIKI_DUMP_LINK=https://dumps.wikimedia.org/${lg}wiki/latest/$WIKI_DUMP_NAME

# install tools
data/wiki/install-tools.sh $TOOLS_PATH

# create Wiki paths
mkdir -p $WIKI_PATH/bz2
mkdir -p $WIKI_PATH/txt

# download Wikipedia dump
if [ ! -f $WIKI_PATH/bz2/enwiki-latest-pages-articles.xml.bz2 ]; then
    echo "Downloading $lg Wikipedia dump from $WIKI_DUMP_LINK ..."
    wget -c $WIKI_DUMP_LINK -P $WIKI_PATH/bz2/
    echo "Downloaded $WIKI_DUMP_NAME in $WIKI_PATH/bz2/$WIKI_DUMP_NAME"
fi

# extract and tokenize Wiki data
#cd $MAIN_PATH
echo "*** Cleaning and tokenizing $lg Wikipedia dump ... ***"
if [ ! -f $WIKI_PATH/txt/$lg.all.raw ]; then
  python $TOOLS_PATH/wikiextractor/WikiExtractor.py $WIKI_PATH/bz2/$WIKI_DUMP_NAME --processes 24 -q -o - \
  | sed "/^\s*\$/d" \
  | grep -v "^<doc id=" \
  | grep -v "</doc>\$" \
  | python $REMOVE_ACCENT \
  > $WIKI_PATH/txt/$lg.all.raw
fi
echo "*** Not Tokenized ( but + accent-removal) $lg Wikipedia dump to $WIKI_PATH/txt/train.${lg} ***"

# split into train / valid / test
echo "*** Split into train / valid / test ***"
split_data() {
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NTRAIN=$((NLINES - 10000));
    NVAL=$((NTRAIN + 5000));
    cat $1 | head -$NTRAIN             > $2;
    cat $1 | head -$NVAL | tail -5000  > $3;
    cat $1 | tail -5000                > $4;
}
split_data $WIKI_PATH/txt/$lg.all.raw $WIKI_PATH/txt/$lg.train.raw $WIKI_PATH/txt/$lg.valid.raw $WIKI_PATH/txt/$lg.test.raw

# File structure
mv $WIKI_PATH/txt/* $WIKI_PATH/
rm -rf $WIKI_PATH/bz2
rm -rf $WIKI_PATH/txt
