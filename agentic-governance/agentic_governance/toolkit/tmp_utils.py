from prettytable import PrettyTable
from prettytable import TableStyle, HRuleStyle


def workflow_table():
    table = PrettyTable()
    table.field_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    table.add_row(
        [
            "",
            "",
            "Questionnaire",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        divider=False,
    )
    table.add_row(["", "", "Prediction", "", "", "", "", "", ""], divider=False)
    table.add_row(
        [
            "Governance",
            "--->",
            "",
            "",
            "--->",
            "Risk Generation",
            "--->",
            "Persist",
            "",
        ],
        divider=False,
    )
    table.add_row(
        ["Orchestrator", "", "", "", "", "Task", "", "Results", ""], divider=False
    )
    table.add_row(["", "", "AI Task", "", "", "", "", "", ""], divider=False)
    table.add_row(["", "", "Identification", "", "", "", "", "", ""], divider=False)
    table.set_style(TableStyle.SINGLE_BORDER)
    table.hrules = HRuleStyle.FRAME
    table.vrules = HRuleStyle.FRAME
    table.header = False
    return table


def workflow_table_2():
    table = PrettyTable()
    table.field_names = ["1", "2", "3", "4", "5", "6"]
    table.add_row(
        [
            "",
            "",
            "Risk",
            "",
            "",
            "",
        ],
        divider=False,
    )
    table.add_row(["", "", "Assessment", "", "", ""], divider=False)
    table.add_row(
        [
            "Governance",
            "--->",
            "",
            "",
            "--->",
            "Incident",
        ],
        divider=False,
    )
    table.add_row(["Orchestrator", "", "", "", "", "Reporting"], divider=False)
    table.add_row(["", "", "Drift", "", "", ""], divider=False)
    table.add_row(["", "", "Monitoring", "", "", ""], divider=False)
    table.set_style(TableStyle.SINGLE_BORDER)
    table.hrules = HRuleStyle.FRAME
    table.vrules = HRuleStyle.FRAME
    table.header = False
    return table
