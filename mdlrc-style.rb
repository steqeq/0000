all
# Extend line length
rule 'MD013', :line_length => 99999

# Allow in-line HTML
exclude_rule 'MD033'

# MyST list-table syntax violates MD004 and MD005
exclude_rule 'MD004'
exclude_rule 'MD005'