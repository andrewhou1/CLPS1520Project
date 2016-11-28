import sys

def read_object_classes(classes_map_filename):
    """
    Reads an index of object classes and their corresponding names and colors.
    Each line of the file has 5 elements: R,G,B values as floats, an integer ID, and a name as a string.
    :param classes_map_filename: The filename storing the index
    :return: a tuple of 4 items:
        1. an array of ID -> category color as RGB tuple (in [0, 255])
        2. a dictionary of category color (as an RGB tuple) -> ID
        3. an array of ID -> category name
        2. a dictionary of category name -> ID
    """
    format_description = "Each line should contain 5 elements: (float R, float G, float B, int ID, str Name)."
    ids = set()
    ids_to_cols = {}
    cols_to_ids = {}
    ids_to_names = {}
    names_to_ids = {}
    with open(classes_map_filename, 'r') as classes_file:
        for line in classes_file:
            try:
                vals = line.split()
                rgb = tuple([int(255 * float(s)) for s in vals[:3]])
                category_num = int(vals[3])
                category_name = vals[4]

                # check for duplicate categories
                if category_num in ids:
                    sys.stderr.write("A category with this number (%d) already exists.\n" % category_num)
                    continue
                if category_name in names_to_ids:
                    sys.stderr.write("A category with this name (%s) already exists.\n" % category_name)
                    continue
                if rgb in cols_to_ids:
                    sys.stderr.write("A category with this color (%s) already exists.\n" % (rgb,))
                    continue

                ids.add(category_num)
                ids_to_names[category_num] = category_name
                names_to_ids[category_name] = category_num
                ids_to_cols[category_num] = rgb
                cols_to_ids[rgb] = category_num

            except (ValueError, IndexError) as e:
                sys.stderr.write("%s %s\n" % (format_description, e), file=sys.stderr)
                continue

    max_id = max(ids)
    category_colors = [None] * (max_id+1)
    category_names = [None] * (max_id+1)
    for cat_id in ids:
        category_names[cat_id] = ids_to_names[cat_id]
        category_colors[cat_id] = ids_to_cols[cat_id]

    return category_colors, cols_to_ids, category_names, names_to_ids


def main():
    filename = sys.argv[1]

    category_colors, cols_to_ids, category_names, names_to_ids = read_object_classes(filename)
    print category_colors
    print cols_to_ids
    print category_names
    print names_to_ids


if __name__ == '__main__':
    main()
