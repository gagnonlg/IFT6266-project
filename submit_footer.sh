# BEGIN FOOTER

cd $_workdir/..
_results=$(basename $_workdir)
tar cpzf $_results.tar.gz $_results
cp $_results.tar.gz $HOME

# END FOOTER
