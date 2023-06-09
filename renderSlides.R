
# To render this document you can proceed as follows:
#   
#   Option 1:
#   - From the terminal tabset, type:
#   
#   quarto render file_to_render.qmd 
# 
# Option 2:
#   - Use the quarto package
# 
# - quarto render file_to_render.qmd  # defaults to html
# - quarto render file_to_render.qmd --to pdf
# 
# Option 3:
#   - Use the renderthis package
#   WARNING: Your antivirus can block the installation
# - First render to html
# 
# to_html(from = "file_to_render.qmd")
# 
# - Then re-render the html file to pdf
# 
# to_pdf(from = "file_to_render.qmd")
# 
# - Compare with:
#   
#   to_pdf(from = "file_to_render.html")

if (!require(quarto))
  install.packages("quarto", dependencies = TRUE)

library(quarto)

quarto::quarto_render(input = "Introduction_to_Deep_Learning.qmd", 
                      output_format = c("html", "pdf"))

quarto::quarto_render(input = "Introduction_to_Deep_Learning-Slides.qmd", 
                      output_format = c("html", "pdf"))

quarto::quarto_render(input = "ConvolutionalNN.qmd", 
                      output_format = c("html", "pdf"))

quarto::quarto_render(input = "ConvolutionalNN-Slides.qmd", 
                      output_format = c("html", "pdf"))

quarto::quarto_render(input = "DNNwithKeras_inR.qmd", 
                      output_format = c("html", "pdf"))



