class FileGraph:
    def __init__(self, filename, df):
        self.filename = filename
        self.df = df
        self.xy_at_peak = None
        self.x_at_target_y = None

    def __repr__(self):
        return f"Filename: {self.filename}\n" + \
            f"\txy at peak: {self.xy_at_peak}\n" + \
            f"\tx at target y: {self.x_at_target_y}\n"