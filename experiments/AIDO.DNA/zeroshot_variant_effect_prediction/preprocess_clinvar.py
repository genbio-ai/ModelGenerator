import pandas as pd
import sys
import os

bed_file_path = sys.argv[1]

# Check if the file exists
if not os.path.isfile(bed_file_path):
    print(f"Error: The file {bed_file_path} does not exist.")
    sys.exit(1)

# https://genome.ucsc.edu/cgi-bin/hgTables?db=hg38&hgta_group=phenDis&hgta_track=clinvar&hgta_table=clinvarMain&hgta_doSchema=describe+table+schema
names = ['chrom', 'start', 'end', 'name', 'score', 'strand',
                'thickStart', 'thickEnd', 'reserved', 'blockCount', 'blockSizes',
                'chromStarts', 'origName', 'clinSign', 'reviewStatus', 'type',
                'geneID', 'molConseq', 'snpID', 'nsvID', 'rcvAcc', 'testedInGtr',
                'phenotypeList', 'phenotype', 'origin', 'assembly', 'cytogenetic',
                '_jsonHgvsTable', '_hgvsProt', 'numSubmit', 'lastEval', 'guidelines',
                'otherIds', '_mouseOver', 'vcfDesc','somImpactDesc','oncogenDesc','_clinSignCode', '_originCode', '_allTypeCode',
                '_varLen', '_starCount', '_variantId', '_dbVarSsvId', 'vcvId']

# Load the BED file
# BED files are typically tab-delimited; the first three columns are mandatory
bed_df = pd.read_csv(bed_file_path, sep='\t', header=None, names = names, engine="python")

## We only want SNPs, and BN and PG
snp_df = bed_df[bed_df["type"] == "single nucleotide variant"]
pg_df = snp_df[snp_df["_clinSignCode"] == "PG"]
bn_df = snp_df[snp_df["_clinSignCode"] == "BN"] ## downsample BN
ds_bn_df = bn_df.sample(n=len(pg_df), random_state=42)
clinvar_df = pd.concat([pg_df, ds_bn_df])\

## selecting only a few columns
df = clinvar_df[["chrom", "start", "end", "name", "_clinSignCode"]]

## name is ref>mut, pull this apart into ref and mut columns
df[['ref', 'mutate']] = df['name'].str.split('>', expand=True)
df['effect'] = df['_clinSignCode'].apply(lambda x: 1 if x == 'PG' else 0)

## make sure that everything is only ATCG and not some weird thing
df_cleaned = df[df["ref"].isin(["A", "T", "C", "G"]) & df["mutate"].isin(["A", "T", "C", "G"])]

## make sure we only wnat these chromosomes, not ChrY
chroms = ['chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
       'chr16', 'chr17', 'chr18', 'chr19', 'chr2', 'chr20', 'chr21',
       'chr22', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9','chrX']

df_cleaned = df_cleaned[df_cleaned["chrom"].isin(chroms)]

savepath = os.path.join(os.path.dirname(bed_file_path), "ClinVar_Processed.tsv")
df_cleaned.to_csv(savepath, sep='\t', index=False)
print(f'save processed file to {savepath}')