"""Public result model and JSON serialization; no reduction work is done here."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .core import INTEGER_ENCODING, Label, Signature
from .forest import InsertionOrigin, serialize_history
from .options import Options
from .orders import ModuleOrder
from .polynomial import IntegerPolynomialRing
from .runtime import environment, native_usage
from .structures import Stats


@dataclass
class Result:
    status: str
    message: str
    backend: IntegerPolynomialRing
    module_order: ModuleOrder
    inputs: list
    basis: list[Label]
    syzygies: list[Signature]
    stats: Stats
    options: Options
    setup_seconds: float = 0.0
    basis_representations: list | None = None
    syzygy_representations: list | None = None
    insertion_history: list[InsertionOrigin | None] | None = None

    generation_paths: object | None = None
    polynomial_postprocessing: object | None = None
    incremental_metadata: dict | None = None

    def to_dict(self, include_basis: bool = True) -> dict:
        answer = {
            "result_schema_version": 2,
            "integer_encoding": INTEGER_ENCODING,
            "status": self.status,
            "message": self.message,
            "variables": list(self.backend.names),
            "coefficient_ring": "ZZ",
            "polynomial_order": self.backend.order_name,
            **self.module_order.metadata(),
            "options": asdict(self.options),
            "environment": environment(),
            "setup_seconds": self.setup_seconds,
            "input_polynomials": [self.backend.serialize(f) for f in self.inputs],
            "basis_size": len(self.basis),
            "syzygy_size": len(self.syzygies),
            "solver_mode": ("incremental-pot" if self.options.incremental_pot else "all-inputs-at-once"),
            "solver_claims_signature_basis": self.status == "completed" and not self.options.incremental_pot,
            "signature_basis_certified": False,
            "certification_scope": "No independent verification has been run by to_dict",
            "representation_certificate_available": self.basis_representations is not None,
            "generation_paths_available": self.generation_paths is not None,
            **self.stats.to_dict(),
        }
        answer["native_usage"] = native_usage(
            answer["counters"], incremental_pot=self.options.incremental_pot)
        if self.options.incremental_pot:
            answer.update(
                solver_claims_strong_basis=self.status == "completed",
                strong_basis_certified=False,
                syzygy_scope="Stage-local syzygies are retired; no global syzygy basis is returned",
                basis_scope="Minimal strong ideal basis; retained labels are not a complete signature basis",
                incremental=dict(self.incremental_metadata or {}),
            )
        if include_basis:
            if self.generation_paths is not None:
                answer["generation_paths"] = self.generation_paths.to_dict()
            if self.insertion_history is not None:
                answer["insertion_history"] = serialize_history(
                    self.insertion_history, len(self.basis), len(self.inputs))
            answer["basis"] = [
                {
                    "signature": label.signature.to_dict(),
                    "polynomial": self.backend.serialize(label.polynomial),
                }
                for label in self.basis
            ]
            answer["syzygy_signatures"] = [signature.to_dict() for signature in self.syzygies]
            if self.basis_representations is not None and self.syzygy_representations is not None:
                answer["representation_certificate"] = {
                    "format": "gvwzz-u-vectors-v1",
                    "basis": [[self.backend.serialize(f) for f in v] for v in self.basis_representations],
                    "syzygies": [[self.backend.serialize(f) for f in v] for v in self.syzygy_representations],
                }
        if self.polynomial_postprocessing is not None:
            answer["polynomial_postprocessing"] = self.polynomial_postprocessing.to_dict(include_basis=include_basis)
        return answer


    def json_basis_fields(self) -> dict:
        """Full exact records prepared lazily for the streaming JSON writer.

        Metadata and verification may be added by the CLI before these fields.
        The public ``to_dict`` API remains an ordinary, eagerly built mapping.
        """
        from .output import JsonSequence

        fields = {}
        if self.generation_paths is not None:
            fields["generation_paths"] = self.generation_paths.to_dict()
        if self.insertion_history is not None:
            fields["insertion_history"] = serialize_history(
                self.insertion_history, len(self.basis), len(self.inputs))
        if not hasattr(self, "_json_source_identity"):
            self._json_source_identity = object()
        fields["basis"] = JsonSequence(lambda: (
            {"signature": label.signature.to_dict(),
             "polynomial": self.backend.serialize(label.polynomial)}
            for label in self.basis), source_identity=self._json_source_identity)
        fields["syzygy_signatures"] = JsonSequence(lambda: (
            signature.to_dict() for signature in self.syzygies))
        if self.basis_representations is not None and self.syzygy_representations is not None:
            def vectors(rows):
                return JsonSequence(lambda: (
                    JsonSequence(lambda row=row: (self.backend.serialize(f) for f in row))
                    for row in rows))
            fields["representation_certificate"] = {
                "format": "gvwzz-u-vectors-v1",
                "basis": vectors(self.basis_representations),
                "syzygies": vectors(self.syzygy_representations),
            }
        return fields

    def to_json_dict(self, include_basis: bool = True) -> dict:
        """Schema-compatible mapping for ``gvwzz.output.write_json`` only."""
        data = self.to_dict(include_basis=False)
        if include_basis:
            data.update(self.json_basis_fields())
        if self.polynomial_postprocessing is not None:
            data["polynomial_postprocessing"] = self.polynomial_postprocessing.to_json_dict(
                include_basis=include_basis)
        return data
