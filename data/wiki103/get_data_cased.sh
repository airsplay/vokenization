OUTPUT=data/wiki103-cased
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip -P $OUTPUT/
unzip $OUTPUT/wikitext-103-raw-v1.zip -d $OUTPUT
mv $OUTPUT/wikitext-103-raw/* $OUTPUT
rm -rf $OUTPUT/wikitext-103-raw-v1.zip $OUTPUT/wikitext-103-raw
