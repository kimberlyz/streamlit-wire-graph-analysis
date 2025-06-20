class FileGraph:
    def __init__(self, filename, df):
        self.filename = filename
        self.df = df
        self.xy_at_peak = None
        self.x_at_target_y = None
        self.y_at_target_x = None

    def __repr__(self):
        return f"Filename: {self.filename}\n" + \
            f"\txy at peak: {self.xy_at_peak}\n" + \
            f"\tx at target y: {self.x_at_target_y}\n" + \
            f"\ty at target x: {self.y_at_target_x}\n"