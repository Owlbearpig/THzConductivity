
def select_measurements(path_list, keywords=None, case_sensitive=True):
    if keywords is None:
        return path_list

    if not case_sensitive:
        keywords = [keyword.lower() for keyword in keywords]
        path_list = [str(path).lower() for path in path_list]

    ret = []
    for filepath in path_list:
        if all([keyword in str(filepath) for keyword in keywords]):
            ret.append(filepath)

    ref_cnt, sam_cnt = 0, 0
    for selected_path in ret:
        if "sam" in str(selected_path).lower():
            sam_cnt += 1
        elif "ref" in str(selected_path).lower():
            ref_cnt += 1
    print(f"Number of reference and sample measurements in selection: {ref_cnt}, {sam_cnt}")

    ret.sort(key=lambda x: x.meas_time)

    print("Time between first and last measurement: ", ret[-1].meas_time - ret[0].meas_time)

    return ret

