"""Bring ModelSEED structures into the forms dGbyG was trained on.

structure_sensitivity.py found two systematic differences between ModelSEED's
SMILES and dGbyG's 643 training structures, neither of which dGbyG's own
normalize_mol reconciles:

  * charged pH-7 forms survive. Given a SMILES, dGbyG adds explicit hydrogens
    before RDKit's Uncharger runs, and with explicit H present the uncharger
    cannot deprotonate a positive nitrogen (so it keeps a balancing negative
    charge too): [NH4+], amino-acid zwitterions, [nH+] adenine in ATP/CoA/NAD.
    The same molecule passed as a Mol WITHOUT explicit H is neutralised.
  * amides are drawn as imidic acids (N=C(O), from ModelSEED's N=C([O-])
    imidate SMILES): glutamine, NAD(H), every CoA amide, dihydroorotate.

The repair ('fix_amide' in structure_sensitivity.py) is: build the dGbyG
Compound from the Mol without explicit H (so it is uncharged properly), then
rewrite every non-aromatic imidic acid / lactim N=C-OH as the amide / lactam
NH-C=O. Aromatic lactims (hydroxypyrimidines) are left alone.

It was designed after inspecting the anchor compounds, so anchor accuracy
measured with it closes the representation gap rather than validating
accuracy; it is reported as a labelled variant, never as the primary estimate.
"""
from rdkit import Chem

IMIDIC = Chem.MolFromSmarts("[#7X2;!a]=[#6X3;!a]-[OX2H1]")


def imidic_to_amide(mol):
    """Rewrite every non-aromatic imidic acid / lactim as the amide / lactam.
    Input and output without explicit H."""
    m = Chem.RWMol(Chem.RemoveHs(mol))
    for _ in range(50):
        hit = m.GetSubstructMatch(IMIDIC)
        if not hit:
            break
        n, c, o = hit
        m.GetBondBetweenAtoms(n, c).SetBondType(Chem.BondType.SINGLE)
        m.GetBondBetweenAtoms(c, o).SetBondType(Chem.BondType.DOUBLE)
        for i in (n, o):
            m.GetAtomWithIdx(i).SetNoImplicit(False)
            m.GetAtomWithIdx(i).SetNumExplicitHs(0)
        Chem.SanitizeMol(m)
    return m.GetMol()


def repaired_compound(api, smiles):
    """dGbyG Compound for a ModelSEED SMILES after the repair; None if unparseable."""
    raw = Chem.MolFromSmiles(smiles)
    if raw is None:
        return None
    fix = api.Compound(Chem.RemoveHs(raw))
    if fix.mol is None:
        return None
    return api.Compound(imidic_to_amide(fix.mol))
