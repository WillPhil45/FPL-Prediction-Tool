import csv

ORIGINAL_CSV = r"Up-to-data\Fantasy-Premier-League\data\2024-25\gws\merged_gw.csv"         # your existing file
CLEANED_CSV  = r"data\merged_gw_clean.csv"   # output file

LINE_THRESHOLD = 14180  # from which line onward we remove columns
START_COL = 21          # 0-based index for "column 22"
END_COL   = 28          # 0-based index for "column 28" (non-inclusive in slicing)

with open(ORIGINAL_CSV, "r", encoding="utf-8", newline='') as infile, \
     open(CLEANED_CSV,  "w", encoding="utf-8", newline='') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    row_index = 0
    for row in reader:
        row_index += 1
        if row_index >= LINE_THRESHOLD:
            # We only remove columns 22..28 if they exist
            # i.e. remove row[21:28] in 0-based indexing
            if len(row) > END_COL:
                # row = row[:21] + row[28:]
                row = row[:START_COL] + row[END_COL:]
        writer.writerow(row)

print(f"Done! Cleaned CSV written to {CLEANED_CSV}")
