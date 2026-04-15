def get_dataanalyser(dataanalyser_type: str):
    if dataanalyser_type == 'ALA2':
        from .ala2 import Ala2DataAnalyser as DataAnalyser
    else:
        raise ValueError("Unknown dataanalyser type: " + dataanalyser_type)
    return DataAnalyser