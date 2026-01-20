suppressMessages(library(dplyr))
suppressMessages(library(tidyr))
suppressMessages(library(ggplot2))
suppressMessages(library(ggVennDiagram))
suppressMessages(library(UpSetR))
suppressMessages(library(ComplexUpset))
suppressMessages(library(argparse))
suppressMessages(library(VennDiagram))

parser <- ArgumentParser()
parser$add_argument("--methods","-m",type="character",help="methods that should be compared, has to be subset of methods in comparison_results.tsv, given as comma-seperated list",default="DESeq2 2-aspect,DESeq2 standard,fuzzy rule,raw log2FC")
parser$add_argument("--compPath","-c",type="character",help="path to the comparison_results.tsv file",default="normalized_results/comparison_results.tsv")
parser$add_argument("--output","-o",type="character",help="path for the outputs, should be a directory")
parser$add_argument("--cmcCut","-cc",type="integer",help="min score to consider mirna as CMC",default=3)
args <- parser$parse_args()
methods<-strsplit(args$methods,split=",")[[1]]
compPath<-args$compPath
out<-args$output
d<-read.table(compPath,header=T,stringsAsFactors=F,sep="\t")
d<-d[is.element(d$method,methods),]
sets<-data.frame(feature=apply(d,1,function(x){paste(x[1:3],collapse="_")}),
					  set=d$method)
cmc<-read.table("gkae017_supplemental_files/known_cancer-specific.tsv",header=T)
cmc<-cmc[cmc$CMC_score>=args$cmcCut,]


draw_my_venn <- function(sets, colors = NULL, filename = NULL) {
  	n_sets <- length(sets)
  	if (n_sets < 2 | n_sets > 4) stop("Only 2-4 sets are supported")
    
	# Automatically compute set names
  	set_names <- names(sets)
  	if (is.null(set_names)) set_names <- paste0("Set", 1:n_sets)
	   # Compute sizes and intersections
	   if (n_sets == 2) {
			s1 <- sets[[1]]; s2 <- sets[[2]]
	     	area1 <- length(s1)
		   area2 <- length(s2)
		   n12 <- length(intersect(s1, s2))
			plot_fn <- draw.pairwise.venn(area1 = area1, area2 = area2, cross.area = n12,
					category = set_names,fill = colors[1:2],, cex = 1.5, cat.cex = 1.5,
					cat.pos = c(0,0), cat.dist = c(0.05,0.05) )
		} else if (n_sets == 3) {
			s1 <- sets[[1]]; s2 <- sets[[2]]; s3 <- sets[[3]]
			plot_fn <- draw.triple.venn(area1 = length(s1), area2 = length(s2), area3 = length(s3),
							n12 = length(intersect(s1,s2)), n23 = length(intersect(s2,s3)),
							n13 = length(intersect(s1,s3)), 
							n123 = length(Reduce(intersect, list(s1,s2,s3))),
							category = set_names, fill = colors[1:3], 
					     cex = 1.5, cat.cex = 1,cat.pos = c(0,0,180), cat.dist = c(0.05,0.05,0.1) )
		} else if (n_sets == 4) {
			s1 <- sets[[1]]; s2 <- sets[[2]]; s3 <- sets[[3]]; s4 <- sets[[4]]
		grid.newpage()
		pushViewport(viewport(width = 0.95, height = 0.95)) 
			plot_fn <- draw.quad.venn( area1 = length(s1), area2 = length(s2), area3 = length(s3), 
							area4 = length(s4), n12 = length(intersect(s1,s2)),
							n13 = length(intersect(s1,s3)), n14 = length(intersect(s1,s4)),
						  	n23 = length(intersect(s2,s3)), n24 = length(intersect(s2,s4)),
							n34 = length(intersect(s3,s4)),n123 = length(Reduce(intersect,list(s1,s2,s3))),
							n124 = length(Reduce(intersect,list(s1,s2,s4))),
							n134 = length(Reduce(intersect,list(s1,s3,s4))),
							n234 = length(Reduce(intersect,list(s2,s3,s4))),
							n1234 = length(Reduce(intersect,list(s1,s2,s3,s4))),
							category = set_names, fill = colors[1:4])
						#	cex = 1.5, cat.cex = 1.5)
		  }
}

visualize<-function(sets, path, prefix){
	upset_df <- sets %>% distinct(feature, set) %>% mutate(value = 1) %>% 
		  pivot_wider(names_from = set,values_from = value, values_fill = 0)
	upset_df2 <- upset_df %>% separate(feature, into = c("mirna", "cluster", "regulation"),
								      sep = "_", remove = FALSE)
	upset_df2 <- upset_df2 %>% left_join( cmc,
									    by = c("mirna" = "feature", "cluster" = "cluster") )
	upset_df2 <- upset_df2 %>% mutate(
		      CMC_class = case_when(is.na(CMC_score)        ~ "not_in_CMC",
								       CMC_score >= args$cmcCut  ~ "CMC" ) )
	set_sizes_df <- upset_df2 %>%
			    select(-feature, -CMC_score, -mirna, -regulation, -cluster) %>%
				   pivot_longer(cols = colnames(upset_df)[-1], names_to = "set", values_to = "in_set") %>% group_by(set, CMC_class) %>%  summarise(n_in_CMC = sum(in_set), .groups = "drop")
	counts<-set_sizes_df %>% group_by(set) %>% summarise(count=sum(n_in_CMC), .groups="drop")
	set_sizes_df$CMC_class<-factor(set_sizes_df$CMC_class,levels=c("not_in_CMC","CMC"))
	upset_df2$CMC_class<-factor(upset_df2$CMC_class,levels=c("not_in_CMC","CMC"))
	set_sizes = ggplot(set_sizes_df, aes(x = factor(set,levels=counts$set[order(counts$count,decreasing=F)]), y = n_in_CMC, fill = CMC_class)) +geom_bar(stat = "identity",width=0.5) +coord_flip() + scale_y_reverse() +theme_minimal() +labs(x = NULL, y = "Number of features")+theme(legend.position="none", axis.text.y = element_blank()) + geom_text(aes(label = n_in_CMC), position = position_stack(vjust = 0.5), color = "black",size = 3 )
	upset = ComplexUpset::upset(upset_df2, intersect = colnames(upset_df)[-1],
							   base_annotations = list("Intersection size" = intersection_size(
								aes(fill = CMC_class))),set_sizes=F)
library(cowplot)
	png(paste0(path,"/",prefix,"_upset.png"))
	print(	plot_grid(		plot_grid(		NULL, set_sizes ,nrow=2,rel_heights=c(1.8,1)), upset, nrow=1, rel_widths = c(1, 3))			)

#	print(upset(as.data.frame(upset_df[, -1]), sets = colnames(upset_df)[-1],order.by = "freq",keep.order = TRUE))
	dev.off()

	png(paste0(path,"/",prefix,"_venn.png"))
	#par(mar=c(6,6,6,2))
	venn_list <- split(sets$feature, sets$set)
	#venn_list<-venn_list[c(3,1,2,4)]
	#p<-ggVennDiagram(venn_list) + scale_fill_gradient(low = "white", high = "steelblue")  +
	#		  theme( plot.margin = margin(t = 20, r = 40, b = 20, l = 40) )
  	#print(p)
	grid.draw(draw_my_venn(venn_list))
  	dev.off()
 }
visualize(sets,out,"all")
for(cancer in unique(d$cluster)){
	dc<-d[d$cluster==cancer,]
	sets<-data.frame(feature=apply(dc,1,function(x){paste(x[1:3],collapse="_")}),set=dc$method)
	visualize(sets,out,cancer)
}
