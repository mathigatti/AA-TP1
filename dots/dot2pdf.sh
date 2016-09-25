#!/bin/bash

for file_name in $(ls *.dot | cut -d"." -f1 )
do
	dot -Tpdf ${file_name}.dot -o ${file_name}.pdf
done