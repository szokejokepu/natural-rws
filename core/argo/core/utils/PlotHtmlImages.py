import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .AbstractPlot import AbstractPlot

#from core.argo.core.utils.argo_utils import make_list, create_list_colors

import numpy as np

import matplotlib.image as mpimg

import pdb

class PlotHtmlImages(AbstractPlot):

    def __init__(self):
        pass

    def create_resource(self, figsize, folders_cartesian_fixed, title):
        with open("core/argo/core/utils/template.html") as f:
            html = f.read()
        html = html.replace("[TITLE]", title)
        return html

    # could be moved in a AbstactMatplotlibPlot
    def save_resource(self, resource, filename):
        print("saving: " + filename + ".html")

        text_file = open(filename + ".html", "wt")
        n = text_file.write(resource)
        text_file.close()

        
    # could be moved in a AbstactPlotImage
    def rows_columns(self, panels, folders_cartesian_fixed):
        n_columns = len(panels)
        n_rows = len(folders_cartesian_fixed)
        return n_rows, n_columns

    def plot_vertical_panel(self,
                            vertical_panels,
                            c,
                            n_rows,
                            n_columns,
                            legend,
                            html,
                            folders_cartesian_fixed,
                            group_by_values,
                            source,
                            degree,
                            string_to_val):

        # only one vertical panel allowed
        assert(len(vertical_panels)==1)
        panel = vertical_panels[0]
        assert(len(panel)==1)
        panel = panel[0]
    
        root_path = "../../../../../"
                    
        num_cols_images = panel["num_cols_images"] # height
        num_rows_images = panel["num_rows_images"] # width
        width = panel["width"]
        height = panel["height"]

        
        head_html = """
	<tr>
	  <td class='tg-0pky' colspan='""" + str(num_cols_images*num_rows_images) + """'>Original images</td>
	</tr>
	<tr>
        """
        css_html = ""
        js_html = """
        function show() {
	   root = '""" + root_path + """';
        """
        
        k = 0

        (source, directory) = folders_cartesian_fixed[0]

        for s in range(num_rows_images):
            for t in range(num_cols_images):

                dir_path = directory + "/" + panel["dirname"]
                
                css_html += """
    .crop-original-img""" + str(k) + """ {
      margin: """ + str(-3*height*s) + """px 0px 0px """ + str(-width*t) + """px;
    }
                """

                head_html += """
	<td class='tg-0pky'>
<div class='crop-original'>
<img id='crop-original-img""" + str(k) + """' class='crop-original-img""" + str(k) + """' src='""" + root_path + dir_path + """/""" + panel["filename"].replace("EPOCH", str(panel["epoch_original"]).zfill(panel["digits"])) + """' title='Original'>
</div>
        </td>
                """
                
                k += 1

        head_html += """
	</tr>
        """

        body_html = ""
        r = 0
                        
        #for panel in vertical_panels:
        #for curve in panel:
        for (source, directory) in sorted(folders_cartesian_fixed):

            #try:

            '''
            if len(group_by_values) == 0:
                max_colors = len(folders_cartesian_fixed)
            else:
                max_colors = len(list(group_by_values.values())[0])

            if max_colors == 0:
                max_colors = 1
            list_colors = create_list_colors(max_colors)
            '''
                
            # file_path = source  + "/" + directory + "/" + curve["filename"]
            dir_path = directory + "/" + panel["dirname"]
            
            #label = self.create_label(curve, directory, legend, source)
            #self.create_plot(marker, color, curve, ax, degree, label, list_colors, x, y)
            #img = mpimg.imread(file_path)

            body_html += """
	<tr>
	  <td colspan=""" + str(num_cols_images*num_rows_images) + """ class='tg-0pky'>
	    <a id='id""" + str(r) + """' href='""" + root_path + dir_path + """'>""" + directory + """</a>
	  </td>
	</tr>  
            """

            body_html += """
	<tr>
            """
            
            #portion_imgs = [] #[None] * num_cols * num_rows

            k = 0
            
            for s in range(num_rows_images):
                for t in range(num_cols_images):
                    #im = img[(margin_y+t*height):(margin_y+(t+1)*height),(margin_x+s*width):(margin_x+(s+1)*width)]
                    #portion_imgs.append(im)

                    body_html += """
	<td class='tg-0pky'>
<div class='crop-rec'>
<img id='crop-model""" + str(r) + """-img""" + str(k) + """' class='crop-model""" + str(r) + """-img""" + str(k) + """' src='' title='Reconstruction'>
</div>
        </td>
                    """

                    css_html += """
    .crop-model""" + str(r) + """-img""" + str(k) + """ {
      margin: """ + str(-2*height -3*height*s) + """px 0px 0px """ + str(-width*t) + """px;
    }
                    """

                    js_html += """
           model = '""" + dir_path + """';
	   model_path = root + '/' + model;
           document.getElementById('epoch').innerHTML = 'Epoch ' + i;
           document.getElementById('crop-model""" + str(r) + """-img""" + str(k) + """').src = model_path + '/""" + panel["filename"] + """'.replace('EPOCH', padNumber(i,""" + str(panel["digits"]) + """));
                    """
                            
                    k +=1
                    
            body_html += """
	</tr>
            """

            ##new_img = np.concatenate(portion_imgs, axis=1)
                
            #n_plots = len(panels)
            #ax = fig.add_subplot(n_rows, n_columns, r * n_columns + c + 1)
            #ax.imshow(new_img)
            
            #if r==0:
            #    ax.set_title(directory.split("/")[-1], loc='left', pad=20)
            #else:
            ##ax.set_title(directory.split("/")[-1], loc='left', pad=0)
            ##ax.get_xaxis().set_visible(False)
            ##ax.get_yaxis().set_visible(False)
                
            r += 1
            
        #except FileNotFoundError as e:
        #    print("FileNotFoundError cannot read file " + file_path + " " + str(e))
            

        '''
        # Options for legend_loc
        # 'best', 'upper right', 'upper left', 'lower left', 'lower right',
        # 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
        loc = panel.get("legend_loc", "upper right")
        bbox_to_anchor = curve.get("bbox_to_anchor", None)
        if bbox_to_anchor is None:
        lgd = ax.legend(loc=loc)
        else:
        lgd = ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)
        '''

        js_html += """
        }
        """
        
        html = html.replace("[WIDTH]", str(width))
        html = html.replace("[HEIGHT]", str(height))
        html = html.replace("[COLSPAN]", str(num_cols_images*num_rows_images))

        
        html = html.replace("[EPOCH]", str(panel["EPOCH"]))
        html = html.replace("[STEP]", str(panel["epoch_step"]))
        html = html.replace("[HEAD]", head_html)
        html = html.replace("[BODY]", body_html)
        html = html.replace("[CSS]", css_html)
        html = html.replace("[JAVASCRIPT]", js_html)
        return html    


        

