#Charlotte B(last edited-11/2)
import numpy as np
import pandas as pd 
import ndex2 as nd 
from pathlib import Path
from scipy.stats import fisher_exact

#parsing the files in the METABRIC dataset 
def load_mutation_data(mut_file):
    df = pd.read_csv(mut_file, sep="\t", skiprows=1, low_memory=False, usecols=["Variant_Classification", "Hugo_Symbol", "Tumor_Sample_Barcode"])
    variant_classification = ["Frame_Shift_Ins", "In_Frame_Ins", "Missense_Mutation", "Silent", "Translation_Start_Site"
                            "Frame_Shift_Del", "In_Frame_Del", "Nonsense_Mutation", "Splice_Site", "Nonstop_Mutation", 
                            "Splice_Region", "Intron", "5'UTR"] 
    df = df[~df["Variant_Classification"].isin(variant_classification)]
    sample_col = "Tumor_Sample_Barcode"
    gene_col = "Hugo_Symbol"
    mat = pd.crosstab(df[sample_col], df[gene_col])
    mat[mat > 0] = 1
    return mat

def load_cna_data(cna_file):
    df = pd.read_csv(cna_file, sep="\t", low_memory=False)
    gene_col = "Hugo_Symbol"
    df.set_index(gene_col, inplace=True)
    if "Entrez_Gene_Id" in df.columns:
        df.drop(columns=["Entrez_Gene_Id"], inplace=True)
    #return only the MB samples x genes, denoted by Hugo_Symbol 
    cna_binary = (df != 0).astype(int).T  
    return cna_binary

def integrate_mutation_cna(mutation_matrix, cna_matrix):
    #get matrix P
    common_genes = mutation_matrix.columns.intersection(cna_matrix.columns)
    common_samples = mutation_matrix.index.intersection(cna_matrix.index)
    mut_sub = mutation_matrix.loc[common_samples, common_genes]
    cna_sub = cna_matrix.loc[common_samples, common_genes]
    combined = (mut_sub | cna_sub).astype(int)
    return combined

def load_sample_to_patient_map(clinical_sample_file):
    #sampleID to PatientID 
    df = pd.read_csv(clinical_sample_file, sep="\t", skiprows=4, usecols=["PATIENT_ID","SAMPLE_ID"])
    return df.set_index("SAMPLE_ID")["PATIENT_ID"]

def load_patient_subtypes(clinical_patient_file):
    #patietID and corresponding subtype dataframe
    df = pd.read_csv(clinical_patient_file, sep="\t", skiprows=4, usecols=["PATIENT_ID", "CLAUDIN_SUBTYPE"])
    return df.set_index("PATIENT_ID")["CLAUDIN_SUBTYPE"]

def get_sample_subtypes(sample_to_patient, patient_subtypes):
    #subtype label to patientID 
    return sample_to_patient.map(patient_subtypes)

def align_labels_with_matrix(sample_subtypes, P0):
    #returns P0 where the samplexgene matches tumor sampleID and Hugo_Symbol
    #retuns labels corresponding to aligned P0
    valid_samples = sample_subtypes.index.intersection(P0.index)
    aligned_labels = sample_subtypes.loc[valid_samples].dropna()
    aligned_P0 = P0.loc[aligned_labels.index]
    return aligned_P0, aligned_labels

#processing for the molecular interaction network using the ndex.bio for 
# the breast cancer protein-protein interaction network 
# The Pathway Commons network was not used due to an issue with the website that has still not been resolved as of 11/25 
def setup_molecular_interaction_network(networkFile, aligned_P0):
    molIN = nd.create_nice_cx_from_file(networkFile)

    node_list = list(molIN.get_nodes())
    nodes = {}
    for _, node in node_list:
        if 'n' in node:
            nodes[node['@id']] = node['n']

    genes = set(nodes.values()) #from network
    pGenes = set(aligned_P0.columns) #from aligned matrix 

    sharedGenes = genes & pGenes 
    missingG = pGenes- genes 
    extraG = genes - pGenes 

    #can take this out later if we want to
    print(f"Shared genes: {len(sharedGenes)}")
    print(f"Genes in your panel but missing in network: {missingG}")
    print(f"Genes in network but not in your panel: {list(extraG)[:10]}")  # show a sample


    edges = []
    edgeList = list(molIN.get_edges())
    for idx, edge in edgeList:
        #name of source node gene and target node gene to define an edge e 
        source = nodes[edge['s']]
        target = nodes[edge['t']]
        # all interactions are regulatory interactions 
        edges.append({"SourceGene": source, 
                    "TargetGene": target})
        
    df_edge = pd.DataFrame(edges, columns=["SourceGene", "TargetGene"])
    print(df_edge.head())

    df_filteredEdges = df_edge[
                        (df_edge["SourceGene"].isin(pGenes)) & 
                        (df_edge["TargetGene"].isin((pGenes))) 
                            ]
    
    return molIN, nodes, df_filteredEdges


# Feature design based on Supplementary Table 1 from Zhang et al 
# only relevant feature were included 
def load_cancer_related_gene_sets(pathway_features, dir):
    #returns the 
    pathway_geneSets = {}
    pathway_info = {}
    
    tsv_files = list(Path(dir).glob('*.tsv'))
    
    if not tsv_files:
        print(f"Error: No .tsv files found in {dir}")
        return pathway_geneSets, pathway_info
    
    print(f"Found {len(tsv_files)} TSV files")
    
    for tsv_file in tsv_files:
        try:
            # Read as two columns: key and value
            df = pd.read_csv(tsv_file, sep='\t', header=None, names=['key', 'value'])
            data = dict(zip(df['key'], df['value']))
            
            pathway_name = data.get('STANDARD_NAME', tsv_file.stem)
            gene_symbols_str = data.get('GENE_SYMBOLS', '')
            
            if gene_symbols_str and pd.notna(gene_symbols_str):
                genes = set(gene_symbols_str.split(','))
                pathway_geneSets[pathway_name] = genes
                
                pathway_info[pathway_name] = {
                    'collection': data.get('COLLECTION', ''),
                    'description': data.get('DESCRIPTION_BRIEF', ''),
                    'num_genes': len(genes),
                    'feature_name': pathway_features.get(pathway_name, pathway_name.lower())
                }
                
            else:
                print(f"{tsv_file.name}: No GENE_SYMBOLS found")
                
        except Exception as e:
            print(f"Error reading {tsv_file.name}: {e}")
    
    print(f"{len(pathway_geneSets)} pathways loaded successfully")
    
    return pathway_geneSets, pathway_info


def compute_pathway_feature(sourceGene, targetGene, pathway_genes):
    # Returns:
    #     0.0 if neither gene in pathway
    #     0.5 if one gene in pathway
    #     1.0 if both genes in pathway
    src = sourceGene in pathway_genes
    tgt = targetGene in pathway_genes
    return (int(src) + int(tgt)) / 2

def extract_edge_attributes(molIN,  df_filteredEdges):    
#since theres no edge scores attached to the network, the mechanism of action or the likelihood score is used 
# all of the genes from this network have regulatory relationships 
    df_edges = df_filteredEdges.copy()
    
    mechanismOfAction_scores = {}
    likelihood_scores = {}
    
    edge_list = list(molIN.get_edges())
    
    for id, _ in edge_list:
        attributes = molIN.get_edge_attributes(id)
        
        for a in attributes: 
            n = a['n']
            v = a['v']
            
            if n == 'Mechanism of Action':
                mechanismOfAction_scores[id] = float(v)
            elif n == 'Likelihood':
                likelihood_scores[id] = float(v)
    
    df_edges['mechanismOfAction_score'] = df_edges.index.map(mechanismOfAction_scores)
    df_edges['likelihood_score'] = df_edges.index.map(likelihood_scores)
    
    print(f"Extracted Mechanism of Action and Likelihood for {len(df_edges)} edges")
    print(f"Mechanism of Action: {df_edges['mechanismOfAction_score'].notna().sum()} values present")
    print(f"Likelihood Score: {df_edges['likelihood_score'].notna().sum()} values present")
    
    return df_edges

def compute_mutation_rates(aligned_P0):
    #num of mutations per gene 
    return aligned_P0.sum() / len(aligned_P0)
    


def compute_subtype_association(aligned_P0, aligned_labels):
    #Association between incidence of mutation and subtypes
    #  returns -log10(p-value) for each gene 
    subtype_pval = {}
    
    unique_subtypes = aligned_labels.unique()
    print(f"Computing subtype association for {len(aligned_P0.columns)} genes across {len(unique_subtypes)} subtypes")
    
    for gene in aligned_P0.columns:
        gene_mutations = aligned_P0[gene].values
        
        min_pval = 1.0  # Start with worst p-value
        
        #check if mutation is associated with subtype 
        for subtype in unique_subtypes:
            subtype_indicator = (aligned_labels == subtype).astype(int).values

            a = np.sum((gene_mutations == 1) & (subtype_indicator == 1))  # mutated & subtype
            b = np.sum((gene_mutations == 1) & (subtype_indicator == 0))  # mutated & !subtype
            c = np.sum((gene_mutations == 0) & (subtype_indicator == 1))  # !mutated & subtype
            d = np.sum((gene_mutations == 0) & (subtype_indicator == 0))  # !mutated & !subtype
            
            # Fisher's exact test
            try:
                _, pval = fisher_exact([[a, b], [c, d]], alternative='two-sided')
                min_pval = min(min_pval, pval)
            except:
                pass
        
        # Convert to -log10(p-value), add small pseudocount to avoid log(0)
        subtype_pval[gene] = -np.log10(min_pval + 1e-300)
    
    result = pd.Series(subtype_pval)
    # print(f"  Min: {result.min():.4f}")
    # print(f"  Max: {result.max():.4f}")
    # print(f"  Mean: {result.mean():.4f}")
    return result

def compute_mutual_exclusivity(aligned_P0, source_gene, target_gene):
    #returns -log10(p-value) for one-sided Fisher's exact test 
    source_mutations = aligned_P0[source_gene].values
    target_mutations = aligned_P0[target_gene].values
    
    a = np.sum((source_mutations == 1) & (target_mutations == 1))  # both mutated
    b = np.sum((source_mutations == 1) & (target_mutations == 0))  # source only
    c = np.sum((source_mutations == 0) & (target_mutations == 1))  # target only
    d = np.sum((source_mutations == 0) & (target_mutations == 0))  # neither 
    
    try:
        _, pval = fisher_exact([[a, b], [c, d]], alternative='less')
        return -np.log10(pval + 1e-300)
    except:
        return 0.0

def edge_feature_matrix(df_edges, mutation_rate, subtype_pval, 
                        pathway_sets, pathway_info, aligned_P0):
    #combine edges and features calculated (47) into 1 mtx to represent the 
    # molecular interaction network used as input in Zhang et al
    # returns finalized MolIn
    
    edge_feature_matrix = df_edges[['SourceGene', 'TargetGene']].copy()
    
    #edge scores 
    edge_feature_matrix['mechanismOfAction_score'] = df_edges['mechanismOfAction_score'].fillna(0.5)
    edge_feature_matrix['likelihood_score'] = df_edges['likelihood_score'].fillna(0.5)
    
    # 26 cancer-related pathway 
    for pathway_name, pathway_genes in pathway_sets.items():
        feature_name = pathway_info[pathway_name]['feature_name']
        edge_feature_matrix[feature_name] = edge_feature_matrix.apply(
            lambda row: compute_pathway_feature(row['SourceGene'], row['TargetGene'], pathway_genes),
            axis=1
        )
    
    #mutation rates if source and taget genes 
    edge_feature_matrix['mutation_rate_source'] = edge_feature_matrix['SourceGene'].map(mutation_rate).fillna(0)
    edge_feature_matrix['mutation_rate_target'] = edge_feature_matrix['TargetGene'].map(mutation_rate).fillna(0)
    
    #mutual exclusivity of mutations to source and target
    edge_feature_matrix['mutual_exclusivity'] = edge_feature_matrix.apply(
        lambda row: compute_mutual_exclusivity(aligned_P0, row['SourceGene'], row['TargetGene']),
        axis=1
    )
    
    #mutation subtype association for source and target 
    edge_feature_matrix['subtype_assoc_source'] = edge_feature_matrix['SourceGene'].map(subtype_pval).fillna(0)
    edge_feature_matrix['subtype_assoc_target'] = edge_feature_matrix['TargetGene'].map(subtype_pval).fillna(0)
    
    #top 5 recurrent genes  
    top5 = ['CCND1', 'ERBB2', 'MYC', 'PIK3CA', 'TP53']
    for gene in top5:
        edge_feature_matrix[f'{gene}_source'] = (edge_feature_matrix['SourceGene'] == gene).astype(int)
        edge_feature_matrix[f'{gene}_target'] = (edge_feature_matrix['TargetGene'] == gene).astype(int)
    
    #self loop 
    edge_feature_matrix['self_loop'] = (edge_feature_matrix['SourceGene'] == edge_feature_matrix['TargetGene']).astype(int)
    
    #fixed intercept 
    edge_feature_matrix['intercept'] = 1
    
    print(f"  Edges: {len(edge_feature_matrix)}")
    print(f"  Features: {len(edge_feature_matrix.columns) - 2}")  # -2 for SourceGene and TargetGene
    
    return edge_feature_matrix


