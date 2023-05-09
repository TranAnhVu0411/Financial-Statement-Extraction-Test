import camelot
import matplotlib.pyplot as plt
import json
from PyPDF2 import PdfReader

img_idx=8
# Get PDF width and height
pdf_page = PdfReader(open("pdf/test{}.pdf".format(img_idx), 'rb')).pages[0]
pdf_shape = pdf_page.mediabox
pdf_height = pdf_shape[3]-pdf_shape[1]
pdf_width = pdf_shape[2]-pdf_shape[0]

# Use camelot to extract table
# Borderless table
tables = camelot.read_pdf(
    "pdf/test{}.pdf".format(img_idx), 
    flavor='stream', 
    edge_tol=1000, 
    row_tol=30, 
    table_areas=['0,{},{},0'.format(int(pdf_height), int(pdf_width))], 
    strip_text='.\n')

# Border table
# tables = camelot.read_pdf("table-extraction-old/pdf/test{}.pdf".format(img_idx), flavor='lattice', flag_size=True)

# Save CSV file 
tables.export('csv/camelot/test{}.csv'.format(img_idx), f='csv', compress=False) # json, excel, html, markdown, sqlite

# Get and Save table cell bounding box metadatas
cell_metadata = []
tablebox_area = []
for idx, table in enumerate(tables):
    tablebox_area.append(table._bbox)
    cell_metadata.append([])
    for row in table.cells:
        for cell in row:
            print(cell)
            cell_metadata[idx].append({'x1': cell.x1, 'y1': cell.y1, 'x2': cell.x2, 'y2': cell.y2})

with open('metadata/metadata{}.json'.format(img_idx), 'w') as f:
    json.dump(cell_metadata, f)

# Visualize table structure
# camelot.plot(tables[0], kind='grid').show()
# plt.show(block=True)
