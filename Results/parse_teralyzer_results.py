import pandas as pd
from consts import teralyzer_result_dir
from imports import *


class TeralyzerResult:
    def __init__(self, filepath):
        self.name = None
        self.filepath = filepath
        self.result_dict = self.parse_file()

    def __str__(self):
        return str(self.filepath)

    def parse_file(self):
        csv_dict = dict(pd.read_csv(self.filepath))
        formatted_dict = {k.replace(" ", ""): v for k, v in csv_dict.items()}

        formatted_dict["freq"] /= THz

        self.name = self.filepath.stem

        return formatted_dict

    def get_n(self):
        n = self.result_dict["ref_ind"] + 1j * self.result_dict["kappa"]
        return array([self.result_dict["freq"], n]).T


def get_all_teralyzer_results():
    found_results = []

    glob = teralyzer_result_dir.glob("**/*")
    for file_path in glob:
        if file_path.is_file() and ".csv" in file_path.name:
            try:
                found_results.append(TeralyzerResult(filepath=file_path))
            except ValueError:
                print(f"Skipping, not correct format: {file_path}")

    return found_results


def select_results(keywords, case_sensitive=True):
    all_results = get_all_teralyzer_results()

    if not case_sensitive:
        keywords = [keyword.lower() for keyword in keywords]

    selected = []
    for result in all_results:
        if all([keyword in str(result) for keyword in keywords]):
            selected.append(result)

    return selected


if __name__ == '__main__':
    results = get_all_teralyzer_results()
    for result in results:
        print(result.name)

plt.show()
