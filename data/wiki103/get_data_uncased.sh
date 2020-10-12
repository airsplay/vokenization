OUTPUT=data/wiki103

wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -P $OUTPUT/
unzip $OUTPUT/wikitext-103-v1.zip -d $OUTPUT
mv $OUTPUT/wikitext-103/* $OUTPUT
rm -rf $OUTPUT/wikitext-103-v1.zip $OUTPUT/wikitext-103
