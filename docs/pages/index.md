import Section from '../components/section'

# Test Project Docs

This site contains documentation for "Test Project."

It's designed to demonstrate the basic layout of our documentation sites, which typically provide documentation for JavaScript or Python projects. Our general approach is to combine MDX pages (for narrative documentation, examples, tutorials, etc.) with pages automatically generated from the code itself using either Sphinx (for Python) or JSDocs (for JavaSCript).

export default ({ children }) => <Section name='intro'>{children}</Section>
